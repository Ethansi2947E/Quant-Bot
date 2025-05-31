"""
Telegram Bot Module for trading notifications and commands.
"""
# Standard library imports
import asyncio
import traceback
from datetime import datetime, UTC
from typing import Dict, Optional
import logging
import os

# Third-party imports
from httpx import ConnectError
from loguru import logger
import telegram
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler

# Local imports
from config.config import TELEGRAM_CONFIG, TRADING_CONFIG

# Singleton instance for global reference
_telegram_bot_instance = None

class TelegramBot:
    def __init__(self, trading_bot=None, config=None):
        """
        Initialize the Telegram bot.
        
        Args:
            trading_bot: Optional reference to the main trading bot
            config: Optional configuration override
        """
        self.trading_bot = trading_bot
        self.config = config or TELEGRAM_CONFIG
        self.trading_config = TRADING_CONFIG
        
        # Get token from config or environment variable
        self.token = self.config.get("token") or os.environ.get("TELEGRAM_TOKEN")
        
        # Check if token is available
        if not self.token:
            logger.error("No Telegram token found in config or environment variables")
            self.token = "dummy_token"  # Placeholder to avoid initialization errors
            
        # Get allowed user IDs
        self.allowed_user_ids = self.config.get("allowed_users", [])
        
        # Convert to strings for easier comparison
        self.allowed_user_ids = [str(id) for id in self.allowed_user_ids]
        
        # Initialize handlers
        self.application = None
        self.keyboard_shown = False
        self.last_update = None
        self.is_running = False
        
        # Initialize command history for each user
        self.command_history = {user_id: [] for user_id in self.allowed_user_ids}
        
        # Initialize trade notification settings
        self.trade_notification_settings = {
            "enabled": self.config.get("trade_notifications", {}).get("enabled", True),
            "open": self.config.get("trade_notifications", {}).get("open", True),
            "close": self.config.get("trade_notifications", {}).get("close", True),
            "signals": self.config.get("trade_notifications", {}).get("signals", False),
            "rejected": self.config.get("trade_notifications", {}).get("rejected", False),
            "tp_hit": self.config.get("trade_notifications", {}).get("tp_hit", True),
            "sl_hit": self.config.get("trade_notifications", {}).get("sl_hit", True),
            "trailing_update": self.config.get("trade_notifications", {}).get("trailing_update", False)
        }
        
        # Check if instance already exists - support singleton pattern
        global _telegram_bot_instance
        if _telegram_bot_instance is not None:
            logger.info("Using existing TelegramBot instance instead of creating a new one")
            return
        
        _telegram_bot_instance = self
        
        # Initialize with config's trading_enabled value if available, otherwise default to True
        self.trading_enabled = True  # Default to enabled
        self.trade_history = []
        self.start_time = datetime.now(UTC)  # Track bot start time
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        self.command_handlers = {}
        self.chat_id = None  # Initialize chat_id attribute
        self.message_counter = 0
        self.last_error_time = None
        self.last_error_message = None
    
    @classmethod
    def get_instance(cls):
        """Return the singleton instance, creating it if it doesn't exist."""
        global _telegram_bot_instance
        if _telegram_bot_instance is None:
            _telegram_bot_instance = cls()
        return _telegram_bot_instance
    
    def is_initialized(self) -> bool:
        """
        Check if the Telegram bot is initialized and running.
        
        Returns:
            bool: True if the bot is initialized and running, False otherwise
        """
        return self.is_running and self.application is not None
    
    async def initialize(self, config):
        """Initialize the Telegram bot with configuration."""
        try:
            # Set Telegram logger level based on main config
            log_level = config.get('LOG_CONFIG', {}).get('level', 'INFO')
            # Convert string level to numeric level
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            # Set the level for all telegram related loggers
            logging.getLogger('telegram').setLevel(numeric_level)
            logging.getLogger('httpx').setLevel(numeric_level)
            logging.getLogger('telegram.ext').setLevel(numeric_level)
            
            logger.info("Starting Telegram bot initialization...")
            
            # If we're already initialized and running, just return
            if self.is_running and self.application is not None:
                logger.info("Telegram bot already initialized and running - skipping initialization")
                return True
                
            self.config = config
            
            # Set trading_enabled from config
            if isinstance(config, dict):
                self.trading_enabled = config.get('trading_enabled', True)
            elif hasattr(config, 'TRADING_CONFIG'):
                self.trading_enabled = config.TRADING_CONFIG.get('trading_enabled', True)
            
            self.start_time = datetime.now(UTC)  # Reset start time on initialization
            
            # Validate bot token
            if not TELEGRAM_CONFIG.get("token"):
                logger.error("No bot token provided in TELEGRAM_CONFIG")
                return False
            
            # Validate user IDs
            if not TELEGRAM_CONFIG.get("allowed_users"):
                logger.error("No allowed user IDs configured!")
                return False
            
            # Convert and validate user IDs
            self.allowed_user_ids = [str(uid) for uid in TELEGRAM_CONFIG["allowed_users"]]
            logger.info(f"Configured for {len(self.allowed_user_ids)} users: {self.allowed_user_ids}")
            
            # Stop any existing application
            await self.stop()
            
            logger.info("Building Telegram application...")
            self.application = Application.builder().token(TELEGRAM_CONFIG["token"]).build()
            
            # Add command handlers with logging
            logger.info("Registering command handlers...")
            handlers = [
                ("start", self.start_command),
                ("enable", self.handle_enable_command),
                ("disable", self.handle_disable_command),
                ("status", self.status_command),
                ("metrics", self.metrics_command),
                ("help", self.help_command),
                ("stop", self.stop_command),
                ("menu", self.show_command_keyboard)  # Add handler for menu command
            ]
            
            for command, handler in handlers:
                self.application.add_handler(CommandHandler(command, handler))
                logger.debug(f"Registered handler for /{command} command")
            
            # Add error handler
            self.application.add_error_handler(self._error_handler)
            
            # Add general message handler for debugging and processing keyboard buttons
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler)
            )
            
            logger.info("Starting Telegram application...")
            
            # Start the bot with retries and exponential backoff
            max_retries = 3
            retry_interval = 5
            
            for attempt in range(max_retries):
                try:
                    await self.application.initialize()
                    await self.application.start()
                    # Check that application.updater exists before accessing start_polling
                    if hasattr(self.application, "updater") and self.application.updater is not None:
                        await self.application.updater.start_polling()
                    else:
                        logger.warning("Application updater is not available, skipping start_polling")
                    self.is_running = True  # Mark as running after successful start
                    break
                except ConnectError as e:
                    logger.error(f"Failed to connect to Telegram API (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying in {retry_interval} seconds...")
                        await asyncio.sleep(retry_interval)
                        retry_interval *= 2  # Exponential backoff
                    else:
                        logger.error("Failed to connect to Telegram API after multiple retries")
                        await self.send_error_alert("Failed to connect to Telegram API. Bot is shutting down.")
                        return False
                except Exception as e:
                    logger.error(f"Error starting Telegram bot (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying in {retry_interval} seconds...")
                        await asyncio.sleep(retry_interval)
                        retry_interval *= 2
                    else:
                        raise
            
            self.bot = self.application.bot
            
            # Verify bot identity
            bot_info = await self.bot.get_me()
            logger.info(f"Bot initialized: @{bot_info.username} (ID: {bot_info.id})")
            
            # Verify users we can connect with without sending a startup message
            successful_users = []
            for user_id in self.allowed_user_ids:
                try:
                    # Just check if we can get chat information about this user
                    # This verifies the bot has permission to talk to this user
                    chat = await self.bot.get_chat(int(user_id))
                    if chat:
                        successful_users.append(user_id)
                        logger.info(f"Successfully verified connection with user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to verify connection with user {user_id}: {str(e)}")
            if not successful_users:
                logger.error("Could not connect with any configured users!")
                return False
            
            # Update allowed users to only those we can actually message
            self.allowed_user_ids = successful_users
            logger.info(f"Successfully initialized with {len(successful_users)} active users")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {str(e)}")
            await self.stop()
            return False
    
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in Telegram updates."""
        logger.error(f"Telegram error: {context.error}")
        try:
            if isinstance(context.error, Exception):
                error_msg = f"""⚠️ <b>Bot Error</b>
Type: {type(context.error).__name__}
Details: {str(context.error)}"""
                
                # Check that update is valid and has a chat ID
                chat_id = None
                if update is not None and hasattr(update, 'effective_chat'):
                    effective_chat = getattr(update, 'effective_chat')
                    if effective_chat is not None and hasattr(effective_chat, 'id'):
                        chat_id = effective_chat.id
                
                # Check that context has a bot
                has_bot = False
                if hasattr(context, 'bot') and context.bot is not None:
                    has_bot = True
                
                if chat_id is not None and has_bot:
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=error_msg,
                            parse_mode='HTML'
                        )
                    except Exception as send_error:
                        logger.error(f"Failed to send error message: {str(send_error)}")
        except (telegram.error.TelegramError, ConnectionError, asyncio.TimeoutError) as e:
            logger.error(f"Error in error handler: {str(e)}")
        except Exception as e:  # Still keep a general catch for unexpected errors
            logger.error(f"Unexpected error in error handler: {str(e)}")
            logger.error(traceback.format_exc())
            
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle incoming messages from users.
        
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            # Store last update for interactive features
            self.last_update = update
            
            # Skip if message is None or user is not authorized
            if not update.message or not update.effective_user:
                logger.debug("Skipping message without message or effective_user")
                return
                
            # Safety check for user ID                
            user_id = update.effective_user.id
            
            # Check if user is authorized
            if str(user_id) not in self.allowed_user_ids:
                logger.warning(f"Unauthorized access from user {user_id}")
                await update.message.reply_text("🔒 Unauthorized access. Your ID has been logged.")
                return
            
            # Mark that the keyboard has been shown to this user
            self.keyboard_shown = True
                
            # Get the text message
            message_text = update.message.text
            
            # Safety check for empty messages
            if not message_text:
                logger.debug("Received empty message")
                await update.message.reply_text("Please send a text message or command.")
                return
            
            # Check for special keyboard options
            if message_text == "⬅️ Back to Main Menu":
                await self.show_command_keyboard(update, context)
                return
                
            # Check if this is a category selection
            category_map = {
                "🔄 Main Controls": "main_controls",
                "📊 Trading Controls": "trading_controls",
                "📈 Performance Analytics": "performance_analytics",
                "🎯 Signal Management": "signal_management",
                "⚙️ Risk & Position Management": "risk_management",
                "🔧 System Settings": "system_settings"
            }
            
            if isinstance(message_text, str) and message_text in category_map:
                category = category_map[message_text]
                await self.show_command_keyboard(update, context, category)
                return
            
            # Check if the message is a command
            if message_text.startswith('/'):
                # This is a keyboard command (e.g. /daily, /profit, etc.)
                # Extract the command without the slash
                command = message_text[1:]
                
                # Handle specific keyboard commands
                if command == 'daily':
                    # Show daily stats summary
                    await update.message.reply_text("📊 <b>Daily Stats Summary</b>\nThis feature is coming soon.", parse_mode='HTML')
                    
                elif command == 'profit':
                    # Show profit summary
                    await self.metrics_command(update, context)
                    
                elif command == 'balance':
                    # Show account balance
                    await update.message.reply_text("💰 <b>Account Balance</b>\nThis feature is coming soon.", parse_mode='HTML')
                    
                elif command == 'status':
                    # Show standard status
                    await self.status_command(update, context)
                    
                elif command == 'status table':
                    # Show status in table format
                    await update.message.reply_text("📈 <b>Status Table</b>\nThis feature is coming soon.", parse_mode='HTML')
                    
                elif command == 'performance':
                    # Show performance metrics
                    await update.message.reply_text("📊 <b>Performance Analysis</b>\nThis feature is coming soon.", parse_mode='HTML')
                    
                elif command == 'count':
                    # Show trade counts
                    await update.message.reply_text("🔢 <b>Trade Counts</b>\nThis feature is coming soon.", parse_mode='HTML')
                    
                elif command == 'start':
                    # Enable trading
                    await self.handle_enable_command(update, context)
                    
                elif command == 'stop':
                    # Disable trading
                    await self.handle_disable_command(update, context)
                    
                elif command == 'help':
                    # Show help
                    await self.help_command(update, context)
                    
                # Check if this is a history command from the keyboard
                elif command == 'history':
                    # Show the history date selection UI
                    await self.show_history_selection(update, context)
                    
                else:
                    # Unrecognized keyboard command
                    await update.message.reply_text(f"Unrecognized command: {message_text}\nUse /menu to see available commands.")
            else:
                # If keyboard not shown yet, inform user to use /start
                if not self.keyboard_shown:
                    await update.message.reply_text(
                        "👋 <b>Welcome to the Trading Bot!</b>\n\n"
                        "To get started, please use the /start command to see the available options.\n\n"
                        "The bot provides a menu-based interface for easy navigation.",
                        parse_mode='HTML'
                    )
                else:
                    # Regular message, show the main menu
                    await update.message.reply_text(
                        "Use /menu to access the command categories, or use /start to begin.",
                        reply_markup=await self.get_command_keyboard()
                    )
                
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            logger.error(traceback.format_exc())

            
    async def get_command_keyboard(self, category=None) -> ReplyKeyboardMarkup:
        """
        Create and return the command keyboard based on the requested category.
        
        Args:
            category: Optional category name to show specific submenu. If None, shows main menu.
            
        Returns:
            ReplyKeyboardMarkup: The appropriate keyboard markup
        """
        # Import at function level to avoid circular imports
        from telegram import KeyboardButton, ReplyKeyboardMarkup
        
        # Define category titles with emojis
        categories = {
            "main_controls": "🔄 Main Controls",
            "trading_controls": "📊 Trading Controls",
            "performance_analytics": "📈 Performance Analytics",
            "signal_management": "🎯 Signal Management",
            "risk_management": "⚙️ Risk & Position Management",
            "system_settings": "🔧 System Settings"
        }
        
        # Define commands for each category
        category_commands = {
            "main_controls": [
                [KeyboardButton("/start"), KeyboardButton("/status"), KeyboardButton("/help")]
            ],
            "trading_controls": [
                [KeyboardButton("/enable"), KeyboardButton("/disable"), KeyboardButton("/stop")]
            ],
            "performance_analytics": [
                [KeyboardButton("/metrics"), KeyboardButton("/history"), KeyboardButton("/daily")],
                [KeyboardButton("/balance"), KeyboardButton("/performance"), KeyboardButton("/statustable")]
            ],
            "signal_management": [
                [KeyboardButton("/listsignalgenerators"), KeyboardButton("/setsignalgenerator")]
            ],
            "risk_management": [
                [KeyboardButton("/enabletrailingstop"), KeyboardButton("/disabletrailingstop")],
                [KeyboardButton("/enablepositionadditions"), KeyboardButton("/disablepositionadditions")]
            ],
            "system_settings": [
                [KeyboardButton("/enablecloseonshutdown"), KeyboardButton("/disablecloseonshutdown")],
                [KeyboardButton("/count"), KeyboardButton("/shutdown")]
            ]
        }
        
        if isinstance(category, str) and category in category_commands and category in categories:
            keyboard = []
            # Add category title
            keyboard.append([KeyboardButton(f"{categories[category]}")])
            # Add category commands
            keyboard.extend(category_commands[category])
            # Add back button
            keyboard.append([KeyboardButton("⬅️ Back to Main Menu")])
            return ReplyKeyboardMarkup(
                keyboard,
                resize_keyboard=True,
                one_time_keyboard=True
            )
        
        # Otherwise return the main menu with categories
        main_menu = []
        main_menu.append([KeyboardButton("📋 COMMAND CATEGORIES 📋")])
        
        # Add each category as a button
        for cat_key, cat_name in categories.items():
            main_menu.append([KeyboardButton(cat_name)])
        
        return ReplyKeyboardMarkup(
            main_menu,
            resize_keyboard=True,
            one_time_keyboard=True
        )
    
    async def show_command_keyboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE, category=None):
        """
        Display the command keyboard to the user.
        This is triggered by the /menu command or category selection.
        
        Args:
            update: Telegram update object
            context: Telegram context object
            category: Optional category name to show specific submenu
        """
        try:
            # Store last update for interactive features
            self.last_update = update
            
            if not update.message or not update.effective_user:
                return
                
            user_id = update.effective_user.id
            
            # Authorization check
            if str(user_id) not in self.allowed_user_ids:
                await update.message.reply_text("Unauthorized access.")
                return
            
            # Get the appropriate keyboard
            keyboard = await self.get_command_keyboard(category)
            
            if category:
                # Category-specific message
                category_title = {
                    "main_controls": "🔄 MAIN CONTROLS",
                    "trading_controls": "📊 TRADING CONTROLS",
                    "performance_analytics": "📈 PERFORMANCE ANALYTICS",
                    "signal_management": "🎯 SIGNAL MANAGEMENT",
                    "risk_management": "⚙️ RISK & POSITION MANAGEMENT",
                    "system_settings": "🔧 SYSTEM SETTINGS"
                }.get(category, "COMMANDS")
                
                await update.message.reply_text(
                    f"<b>{category_title}</b>\n\n"
                    f"Select a command or go back to the main menu:",
                    reply_markup=keyboard,
                    parse_mode='HTML'
                )
            else:
                # Main menu message
                await update.message.reply_text(
                    "📱 <b>Trading Bot Command Menu</b>\n\n"
                    "Please select a category to see available commands:\n\n"
                    "• 🔄 <b>Main Controls</b>: Basic bot controls\n"
                    "• 📊 <b>Trading Controls</b>: Enable/disable trading\n"
                    "• 📈 <b>Performance Analytics</b>: View trading results\n"
                    "• 🎯 <b>Signal Management</b>: Manage trading signals\n"
                    "• ⚙️ <b>Risk & Position Management</b>: Control risk settings\n"
                    "• 🔧 <b>System Settings</b>: Configure system behavior\n",
                    reply_markup=keyboard,
                    parse_mode='HTML'
                )
            
            # Mark that we've shown the keyboard
            self.keyboard_shown = True
            
        except Exception as e:
            logger.error(f"Error showing command keyboard: {str(e)}")
            logger.error(traceback.format_exc())
            if update and update.message:
                await update.message.reply_text(f"Error showing command menu: {str(e)}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        try:
            # Store last update for interactive features
            self.last_update = update
            
            if not update.message or not update.effective_user:
                return
                
            user_id = update.effective_user.id
            
            # Authorization check
            if str(user_id) not in self.allowed_user_ids:
                await update.message.reply_text("Unauthorized access.")
                return
            
            # Welcome message with keyboard
            await update.message.reply_text(
                "👋 <b>Welcome to the Trading Bot!</b>\n\n"
                "I'm your assistant for managing the trading system. "
                "The commands are organized into categories for easier navigation.\n\n"
                "Select a category below to see available commands:\n",
                reply_markup=await self.get_command_keyboard(),
                parse_mode='HTML'
            )
            
            # Mark that we've shown the keyboard
            self.keyboard_shown = True
            
        except Exception as e:
            logger.error(f"Error in start command: {str(e)}")
            if update and update.message:
                await update.message.reply_text(f"Error: {str(e)}")
    
    async def enable_trading_core(self):
        """Enable trading."""
        try:
            self.trading_enabled = True
            # Update config if available
            if hasattr(self, 'config'):
                if isinstance(self.config, dict):
                    self.config['trading_enabled'] = True
                elif hasattr(self.config, 'TRADING_CONFIG'):
                    self.config.TRADING_CONFIG['trading_enabled'] = True
            return True
        except Exception as e:
            logger.error(f"Error enabling trading: {str(e)}")
            return False

    async def disable_trading_core(self):
        """Disable trading."""
        try:
            self.trading_enabled = False
            # Update config if available
            if hasattr(self, 'config'):
                if isinstance(self.config, dict):
                    self.config['trading_enabled'] = False
                elif hasattr(self.config, 'TRADING_CONFIG'):
                    self.config.TRADING_CONFIG['trading_enabled'] = False
            return True
        except Exception as e:
            logger.error(f"Error disabling trading: {str(e)}")
            return False

    async def handle_enable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
        """
        Handle the /enable command to enable trading.
        
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            # Skip if message or effective_user is None
            if not update or not update.effective_user or not update.message:
                logger.warning("Skipping enable command due to missing update components")
                return
                
            user_id = update.effective_user.id
            
            # Authorization check
            if str(user_id) not in self.allowed_user_ids:
                if update.message:
                    await update.message.reply_text("Unauthorized access.")
                return
                
            logger.info("Enabling trading via Telegram command")
            
            # Get current enabled status
            current_status = self.trading_enabled
            
            if current_status:
                if update.message:
                    await update.message.reply_text("✅ Trading is already enabled.")
                return
                
            # Enable trading in the core
            await self.enable_trading_core()
            
            # Log and notify user
            logger.info(f"Trading enabled by user {update.effective_user.username} (ID: {user_id})")
            if update.message:
                await update.message.reply_text("✅ Trading has been enabled.")
        except Exception as e:
            logger.error(f"Error in enable command: {str(e)}")
            logger.error(traceback.format_exc())
            if update and update.message:
                await update.message.reply_text(f"❌ Error enabling trading: {str(e)}")
                
    async def handle_disable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
        """
        Handle the /disable command to disable trading.
        
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            # Skip if message or effective_user is None
            if not update or not update.effective_user or not update.message:
                logger.warning("Skipping disable command due to missing update components")
                return
                
            user_id = update.effective_user.id
            
            # Authorization check
            if str(user_id) not in self.allowed_user_ids:
                if update.message:
                    await update.message.reply_text("Unauthorized access.")
                return
                
            logger.info("Disabling trading via Telegram command")
            
            # Get current enabled status
            current_status = self.trading_enabled
            
            if not current_status:
                if update.message:
                    await update.message.reply_text("❌ Trading is already disabled.")
                return
                
            # Disable trading in the core
            await self.disable_trading_core()
            
            # Log and notify user
            logger.info(f"Trading disabled by user {update.effective_user.username if update.effective_user else 'Unknown'} (ID: {user_id})")
            if update.message:
                await update.message.reply_text("❌ Trading has been disabled.")
        except Exception as e:
            logger.error(f"Error in disable command: {str(e)}")
            logger.error(traceback.format_exc())
            if update and update.message:
                await update.message.reply_text(f"❌ Error disabling trading: {str(e)}")
                
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
        """Handle the /status command to get bot status."""
        if not update or not hasattr(update, 'effective_user') or update.effective_user is None:
            logger.warning("Received invalid update in status_command")
            return
            
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_users"]:
            try:
                # Get current status
                status = "enabled" if self.trading_enabled else "disabled"
                status_emoji = "✅" if self.trading_enabled else "❌"
                
                # Get bot info if available
                bot_info = None
                if self.bot:
                    try:
                        bot_info = await self.bot.get_me()
                    except Exception as e:
                        logger.error(f"Error getting bot info: {str(e)}")
                
                # Create status message
                status_text = f"""<b>🤖 BOT STATUS</b>

<b>Trading:</b> {status_emoji} <b>{status.upper()}</b>
<b>Bot Online:</b> {'✅ YES' if self.is_running else '❌ NO'}"""

                # Add bot info if available
                if bot_info:
                    status_text += f"\n<b>Bot Name:</b> @{bot_info.username}"
                
                # Add uptime if available
                if hasattr(self, 'start_time'):
                    uptime = datetime.now(UTC) - self.start_time
                    days = uptime.days
                    hours, remainder = divmod(uptime.seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
                    status_text += f"\n<b>Uptime:</b> {uptime_str}"
                
                status_text += f"\n\n<i>Updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
                
                if update.message and hasattr(update.message, 'reply_text'):
                    await update.message.reply_text(status_text, parse_mode='HTML')
            except Exception as e:
                logger.error(f"Error generating status: {str(e)}")
                if update.message and hasattr(update.message, 'reply_text'):
                    await update.message.reply_text(f"Error retrieving status: {str(e)[:100]}")
        else:
            if update.message and hasattr(update.message, 'reply_text'):
                await update.message.reply_text("Unauthorized access.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
        """Handle the /help command to show available commands."""
        if not update or not hasattr(update, 'effective_user') or update.effective_user is None:
            logger.warning("Received invalid update in help_command")
            return
            
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_users"]:
            help_text = """<b>📱 TRADING BOT COMMANDS 📱</b>

<b>➡️ BASIC CONTROLS</b>
/start - Start the bot
/status - Check bot status
/help - Show this help menu

<b>➡️ TRADING CONTROLS</b>
/enable - Enable trading
/disable - Disable trading
/stop - Stop bot and close all trades

<b>➡️ PERFORMANCE DATA</b>
/metrics - View detailed performance stats
/history - View recent trade history

<b>➡️ ADVANCED SETTINGS</b>
/listsignalgenerators - Show available signal generators
/setsignalgenerator - Change signal generator
/enabletrailing - Enable trailing stop loss
/disabletrailing - Disable trailing stop loss
/enablepositionadditions - Allow adding to positions
/disablepositionadditions - Disable adding to positions

<b>➡️ SYSTEM COMMANDS</b>
/enablecloseonshutdown - Enable closing positions on shutdown
/disablecloseonshutdown - Disable closing positions on shutdown
/shutdown - Request graceful shutdown

<i>All commands are restricted to authorized users only</i>"""
            
            if update.message and hasattr(update.message, 'reply_text'):
                await update.message.reply_text(help_text, parse_mode='HTML')
        else:
            if update.message and hasattr(update.message, 'reply_text'):
                await update.message.reply_text("Unauthorized access.")

    async def metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
        """Handle the /metrics command to show performance metrics."""
        if not update or not hasattr(update, 'effective_user') or update.effective_user is None:
            logger.warning("Received invalid update in metrics_command")
            return
            
        user_id = update.effective_user.id
        # Use the bot's allowed_user_ids (already converted to strings)
        if str(user_id) in self.allowed_user_ids:
            try:
                # Calculate performance stats with safety checks
                total_trades = self.performance_metrics.get('total_trades', 0)
                
                # Check if there are any trades
                if total_trades == 0:
                    if update.message and hasattr(update.message, 'reply_text'):
                        await update.message.reply_text(
                            "📊 <b>No trading data available yet</b>\n\nMetrics will appear after completed trades.", 
                            parse_mode='HTML'
                        )
                    return
                
                # Extract metrics safely with defaults
                winning_trades = self.performance_metrics.get('winning_trades', 0)
                losing_trades = self.performance_metrics.get('losing_trades', 0)
                profit = self.performance_metrics.get('total_profit', 0.0)
                max_dd = self.performance_metrics.get('max_drawdown', 0.0)
                
                # Calculate win rate and other derived metrics
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                avg_profit_per_trade = profit / total_trades if total_trades > 0 else 0
                
                # Create visual win rate bar
                if win_rate >= 80:
                    win_rate_visual = "🟩🟩🟩🟩🟩"
                    performance_emoji = "🔥"
                elif win_rate >= 60:
                    win_rate_visual = "🟩🟩🟩🟩⬜"
                    performance_emoji = "📈"
                elif win_rate >= 40:
                    win_rate_visual = "🟩🟩🟩⬜⬜"
                    performance_emoji = "📊"
                elif win_rate >= 20:
                    win_rate_visual = "🟩🟩⬜⬜⬜"
                    performance_emoji = "📉"
                else:
                    win_rate_visual = "🟩⬜⬜⬜⬜"
                    performance_emoji = "❄️"
                
                # Create profit indicator
                profit_indicator = "📈" if profit > 0 else "📉" if profit < 0 else "➖"
                
                metrics_text = f"""{performance_emoji} <b>PERFORMANCE SUMMARY</b> {performance_emoji}

<b>📊 Trade Statistics:</b>
• Total Trades: <b>{total_trades}</b>
• Winning: <b>{winning_trades}</b> | Losing: <b>{losing_trades}</b>
• Win Rate: <b>{win_rate:.1f}%</b> {win_rate_visual}

<b>💰 Profit Analysis:</b>
• Total P/L: <b>{profit_indicator} {profit:.2f}</b>
• Avg Per Trade: <b>{avg_profit_per_trade:.2f}</b>
• Max Drawdown: <b>{max_dd:.2f}%</b>

<i>Updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"""
                
                if update.message and hasattr(update.message, 'reply_text'):
                    await update.message.reply_text(metrics_text, parse_mode='HTML')
            
            except Exception as e:
                logger.error(f"Error generating metrics: {str(e)}")
                logger.error(traceback.format_exc())
                if update.message and hasattr(update.message, 'reply_text'):
                    await update.message.reply_text(
                        f"⚠️ <b>Error retrieving metrics</b>\n\nPlease try again later. Error: {str(e)[:100]}...", 
                        parse_mode='HTML'
                    )
        else:
            if update.message and hasattr(update.message, 'reply_text'):
                await update.message.reply_text("Unauthorized access.")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
        """Handle the /stop command to stop the bot."""
        if not update or not hasattr(update, 'effective_user') or update.effective_user is None:
            logger.warning("Received invalid update in stop_command")
            return
            
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_users"]:
            self.trading_enabled = False
            # Signal to close all trades
            await self.send_message("⚠️ Stopping bot and closing all trades...")
            # The actual trade closing logic should be handled by the main trading bot
            if update.message and hasattr(update.message, 'reply_text'):
                await update.message.reply_text("Bot stopped and all trades are being closed.")
        else:
            if update.message and hasattr(update.message, 'reply_text'):
                await update.message.reply_text("Unauthorized access.")
    
    async def send_setup_alert(self, symbol: str, timeframe: str, setup_type: str, confidence: float):
        """Send setup formation alert to users."""
        if self.bot:
            # Format confidence with stars
            conf_pct = int(confidence * 100)
            if conf_pct >= 80:
                conf_indicator = "⭐⭐⭐⭐⭐"
            elif conf_pct >= 60:
                conf_indicator = "⭐⭐⭐⭐"
            elif conf_pct >= 40:
                conf_indicator = "⭐⭐⭐"
            elif conf_pct >= 20:
                conf_indicator = "⭐⭐"
            else:
                conf_indicator = "⭐"
                
            # Determine setup type emoji
            if any(keyword in setup_type.lower() for keyword in ['bullish', 'buy', 'long']):
                setup_emoji = "🟢"
            elif any(keyword in setup_type.lower() for keyword in ['bearish', 'sell', 'short']):
                setup_emoji = "🔴"
            else:
                setup_emoji = "🔍"
            
            alert_msg = f"""🎯 <b>SETUP DETECTED</b> 🎯

<b>Instrument:</b> {symbol}
<b>Timeframe:</b> {timeframe}
<b>Pattern:</b> {setup_emoji} {setup_type}
<b>Confidence:</b> {conf_indicator} ({conf_pct}%)

<i>This setup may lead to a trading opportunity soon. Monitor price action for confirmation.</i>

⏰ <i>{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"""
            
            await self.send_message(alert_msg)

    async def send_management_alert(self, message: str, alert_type: str = "info") -> bool:
        """Send trade management alert message."""
        try:
            # Check if bot is running
            if not self.is_running or not self.bot:
                logger.warning("Cannot send management alert - Telegram bot not running")
                # Don't try to initialize here
                return False
                
            # Format message based on alert type
            if alert_type.lower() == "warning":
                emoji = "⚠️"
                title = "WARNING"
            elif alert_type.lower() == "error":
                emoji = "🚫"
                title = "ERROR"
            elif alert_type.lower() == "success":
                emoji = "✅"
                title = "SUCCESS"
            else:
                emoji = "ℹ️"
                title = "INFO"
            
            # Format with timestamp
            timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
            
            formatted_message = f"""{emoji} <b>MANAGEMENT ALERT: {title}</b>

{message}

<i>{timestamp} UTC</i>"""
            
            # Send to all users if self.chat_id is not defined
            if not hasattr(self, 'chat_id') or self.chat_id is None:
                # Send to all allowed users
                success = False
                for user_id in self.allowed_user_ids:
                    try:
                        await self.bot.send_message(
                            chat_id=int(user_id),
                            text=formatted_message,
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                        success = True
                        logger.debug(f"Sent management alert to user {user_id}")
                    except Exception as e:
                        logger.error(f"Failed to send management alert to {user_id}: {str(e)}")
                
                return success
            
            # Send to specific chat_id
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                return True
            except Exception as e:
                logger.error(f"Failed to send management alert to {self.chat_id}: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending management alert: {str(e)}")
            return False

    def update_metrics(self, trade_result: Dict):
        """Update performance metrics with new trade result."""
        try:
            logger.info(f"Updating metrics with trade result: {trade_result}")
            
            # Validate trade result data
            if 'pnl' not in trade_result:
                logger.error("Missing PnL in trade result, cannot update metrics")
                return
                
            # Update trade counts
            self.performance_metrics['total_trades'] += 1
            
            # Update win/loss counters
            if trade_result['pnl'] > 0:
                self.performance_metrics['winning_trades'] += 1
                logger.info(f"Added winning trade, new count: {self.performance_metrics['winning_trades']}")
            else:
                self.performance_metrics['losing_trades'] += 1
                logger.info(f"Added losing trade, new count: {self.performance_metrics['losing_trades']}")
            
            # Update profit
            previous_profit = self.performance_metrics['total_profit']
            self.performance_metrics['total_profit'] += trade_result['pnl']
            logger.info(f"Updated total profit: {previous_profit} -> {self.performance_metrics['total_profit']}")
            
            # Update max drawdown (simplified calculation)
            if trade_result['pnl'] < 0 and abs(trade_result['pnl']) > self.performance_metrics['max_drawdown']:
                self.performance_metrics['max_drawdown'] = abs(trade_result['pnl'])
                logger.info(f"Updated max drawdown: {self.performance_metrics['max_drawdown']}")
            
            # Add trade to history with all required fields
            trade_entry = {
                'id': trade_result.get('id', f"trade_{len(self.trade_history) + 1}"),
                'symbol': trade_result.get('symbol', 'Unknown'),
                'type': trade_result.get('type', 'Unknown'),
                'entry': trade_result.get('entry', 0.0),
                'exit_price': trade_result.get('exit', 0.0),
                'pnl': trade_result['pnl'],
                'time': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.trade_history.append(trade_entry)
            logger.info(f"Added trade to history, now have {len(self.trade_history)} trades")
            
            # Keep only last 100 trades in history
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
                
            # Verify metrics consistency
            if self.performance_metrics['winning_trades'] + self.performance_metrics['losing_trades'] != self.performance_metrics['total_trades']:
                logger.warning("Metrics inconsistency detected: winning + losing != total trades")
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            logger.error(traceback.format_exc())

    async def send_performance_update(
        self,
        chat_id: int,
        total_trades: int,
        winning_trades: int,
        total_profit: float
    ):
        """Send a performance update."""
        if self.bot is None:
            logger.error("Cannot send performance update: bot is not initialized")
            return
            
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        message = f"""📊 <b>Performance Update</b>

Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {int(win_rate)}%
Total Profit: {total_profit:.2f}

Keep up the good work! 📈"""
        await self.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='HTML'
        )

    async def send_trade_error_alert(
        self,
        symbol: str,
        error_type: str,
        details: str,
        retry_count: int = 0,
        additional_info: Optional[Dict] = None
    ):
        """Send detailed trade execution error alert to users."""
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return

        # Format error message
        error_msg = f"""⚠️ <b>Trade Execution Error</b>

Symbol: {symbol}
Error Type: {error_type}
Details: {details}
Retry Attempts: {retry_count}"""

        # Use empty dict if additional_info is None
        safe_additional_info = additional_info or {}

        # Add additional info if available
        if safe_additional_info:
            error_msg += "\n\nAdditional Information:"
            for key, value in safe_additional_info.items():
                error_msg += f"\n{key}: {value}"

        error_msg += f"\n\nTime: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"

        # Send with enhanced error handling
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with asyncio.timeout(15):  # 15 second timeout
                    for user_id in self.allowed_user_ids:
                        try:
                            await self.bot.send_message(
                                chat_id=int(user_id),
                                text=error_msg,
                                parse_mode='HTML',
                                disable_web_page_preview=True,  # Speed up message sending
                                disable_notification=False  # Enable notifications for trade errors
                            )
                            logger.info(f"Sent trade error alert to user {user_id}")
                        except Exception as e:
                            logger.error(f"Failed to send trade error alert to user {user_id}: {str(e)}")
                    return
                    
            except asyncio.TimeoutError:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Timeout sending trade error alert (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    
            except telegram.error.RetryAfter as e:
                logger.warning(f"Rate limited, waiting {e.retry_after} seconds")
                await asyncio.sleep(e.retry_after)
                continue
                
            except Exception as e:
                logger.error(f"Error sending trade error alert: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                else:
                    logger.error("Failed to send trade error alert after all retries")

    async def send_trade_alert(
        self,
        chat_id: int,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        confidence: float,
        reason: str
    ):
        """Send a trade alert to the specified user."""
        try:
            if not self.is_running or self.bot is None:
                logger.warning("Cannot send trade alert: bot is not running or initialized")
                # Don't try to initialize here
                return

            # Format the trade alert message
            alert_message = self.format_alert(
                symbol=symbol,
                direction=direction,
                entry=entry,
                sl=sl,
                tp=tp,
                confidence=confidence,
                reason=reason
            )
            
            # Send the alert to the user
            if str(chat_id) in self.allowed_user_ids:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=alert_message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
            else:
                # If specific chat_id not allowed, send to all allowed users
                for user_id in self.allowed_user_ids:
                    try:
                        await self.bot.send_message(
                            chat_id=int(user_id),
                            text=alert_message,
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                    except Exception as e:
                        logger.error(f"Failed to send trade alert to user {user_id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error sending trade alert: {str(e)}")

    def format_alert(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        confidence: float,
        reason: str
    ) -> str:
        """Format a trade alert message."""
        # Calculate risk-reward ratio
        if sl != 0 and entry != 0:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        else:
            rr_ratio = 0
            
        # Format direction with emoji
        if direction.lower() == "buy" or direction.lower() == "long":
            direction_emoji = "🟢 BUY/LONG"
        elif direction.lower() == "sell" or direction.lower() == "short":
            direction_emoji = "🔴 SELL/SHORT"
        else:
            direction_emoji = direction
            
        # Calculate percentage for stop loss and take profit
        if entry != 0:
            sl_percent = ((sl - entry) / entry) * 100
            tp_percent = ((tp - entry) / entry) * 100
        else:
            sl_percent = 0
            tp_percent = 0
            
        # Format confidence with stars
        conf_pct = int(confidence * 100)
        if conf_pct >= 80:
            conf_indicator = "⭐⭐⭐⭐⭐"
        elif conf_pct >= 60:
            conf_indicator = "⭐⭐⭐⭐"
        elif conf_pct >= 40:
            conf_indicator = "⭐⭐⭐"
        elif conf_pct >= 20:
            conf_indicator = "⭐⭐"
        else:
            conf_indicator = "⭐"
            
        return f"""📊 <b>TRADE SIGNAL: {symbol}</b> 📊

<b>{direction_emoji}</b> at <b>{entry:.5f}</b>

<b>Key Levels:</b>
📉 Stop Loss: <b>{sl:.5f}</b> ({sl_percent:.2f}%)
📈 Take Profit: <b>{tp:.5f}</b> ({tp_percent:.2f}%)
⚖️ Risk/Reward: <b>{rr_ratio}:1</b>

<b>Signal Quality:</b> {conf_indicator} ({conf_pct}%)

<b>Analysis:</b>
{reason}

⏰ <i>{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
⚠️ <i>Trade at your own risk - Apply proper risk management</i>"""

    async def notify_error(self, chat_id: int, error: str):
        """Send an error notification."""
        # Get current timestamp
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine error severity by checking for common keywords
        if any(keyword in error.lower() for keyword in ['critical', 'fatal', 'crash', 'exception']):
            severity_emoji = "🚨"
            severity_text = "CRITICAL ERROR"
        elif any(keyword in error.lower() for keyword in ['fail', 'error', 'invalid']):
            severity_emoji = "⚠️"
            severity_text = "ERROR"
        else:
            severity_emoji = "ℹ️"
            severity_text = "WARNING"
            
        error_message = f"""{severity_emoji} <b>{severity_text}</b> {severity_emoji}

<b>Details:</b>
{error}

<b>Time:</b> <i>{timestamp} UTC</i>

<i>Please check the logs for more information. If this issue persists, you may need to restart the bot or check your configuration.</i>"""

        try:
            if self.bot is not None:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=error_message,
                    parse_mode='HTML',
                    disable_notification=False  # Important errors should trigger notifications
                )
                logger.info(f"Sent error notification to user {chat_id}")
            else:
                logger.error("Cannot send error notification: bot is not initialized")
        except Exception as e:
            logger.error(f"Failed to send error notification: {str(e)}")

    async def notify_performance(self, chat_id: str, data: Dict):
        """Send performance update to specified chat."""
        try:
            # Validate input data
            if 'total_trades' not in data or data['total_trades'] == 0:
                logger.warning("Performance update requested with no trade data")
                return
                
            # Calculate performance metrics
            total_trades = data.get('total_trades', 0)
            winning_trades = data.get('winning_trades', 0)
            profit = data.get('profit', 0.0)
            
            # Calculate win rate
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Determine performance emoji
            if win_rate >= 70 and profit > 0:
                performance_emoji = "🔥"
            elif win_rate >= 50 and profit > 0:
                performance_emoji = "📈"
            elif profit > 0:
                performance_emoji = "✅"
            elif profit < 0:
                performance_emoji = "📉"
            else:
                performance_emoji = "➖"
                
            # Format profit with sign
            profit_sign = "+" if profit > 0 else ""
            
            message = f"""{performance_emoji} <b>PERFORMANCE UPDATE</b> {performance_emoji}

<b>📊 Trade Statistics:</b>
• Total Trades: <b>{total_trades}</b>
• Winning Trades: <b>{winning_trades}</b>
• Win Rate: <b>{win_rate:.1f}%</b>
• Total P/L: <b>{profit_sign}{profit:.2f}</b>

<i>Updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"""

            if self.bot:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                logger.info(f"Sent performance update to {chat_id}")
        except Exception as e:
            logger.error(f"Error sending performance update: {str(e)}")
            logger.error(traceback.format_exc())

    async def process_command(self, message):
        """Process a command message."""
        if message.text == "/start":
            welcome_message = """👋 <b>Welcome to Trading Bot!</b>

Thank you for using our service. Use /help to see available commands.

Stay profitable! 📈"""
            if self.bot is not None:
                await self.bot.send_message(
                    chat_id=message.chat.id,
                    text=welcome_message,
                    parse_mode='HTML'
                )
            else:
                logger.error("Cannot process command: bot is not initialized")

    async def check_auth(self, chat_id: int) -> bool:
        """Check if a user is authorized."""
        return str(chat_id) in self.allowed_user_ids
    
    async def send_error_alert(self, message: str) -> bool:
        """Send error alert to all allowed users."""
        if not self.is_running or not self.bot:
            logger.warning("Cannot send error alert: bot is not running")
            return False
            
        success = False
        for user_id in self.allowed_user_ids:
            try:
                await self.bot.send_message(
                    chat_id=int(user_id),
                    text=f"❌ ERROR: {message}",
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
                success = True
            except Exception as e:
                logger.error(f"Failed to send error alert to {user_id}: {str(e)}")
        return success
        
    async def start(self):
        """Start the Telegram bot if not already running."""
        if self.is_running:
            logger.info("Telegram bot is already running")
            return
            
        if not self.config:
            logger.warning("No config set for Telegram bot, initializing with default")
            await self.initialize({})
        else:
            await self.initialize(self.config)
        
        return
        
    async def stop(self):
        """Stop the Telegram bot."""
        try:
            if self.is_running:
                logger.info("Stopping Telegram bot...")
                
                # First mark bot as stopping to prevent new operations
                self.is_running = False
                self.trading_enabled = False
                
                # Create a flag to track successful shutdown
                shutdown_successful = False
                
                try:
                    # Try graceful shutdown of the updater first (with a timeout)
                    if hasattr(self, 'application') and self.application:
                        if hasattr(self.application, 'updater') and self.application.updater:
                            try:
                                # Set a timeout for updater shutdown
                                shutdown_task = asyncio.create_task(self.application.updater.stop())
                                try:
                                    # Wait with a timeout
                                    await asyncio.wait_for(shutdown_task, timeout=5.0)
                                    logger.info("Updater stopped successfully")
                                except asyncio.TimeoutError:
                                    logger.warning("Updater shutdown timed out, proceeding with force stop")
                            except Exception as e:
                                logger.warning(f"Error stopping updater: {str(e)}")
                        
                        # Then shutdown the application (with a timeout)
                        try:
                            # Set a timeout for application shutdown
                            shutdown_task = asyncio.create_task(self.application.shutdown())
                            try:
                                # Wait with a timeout
                                await asyncio.wait_for(shutdown_task, timeout=5.0)
                                shutdown_successful = True
                                logger.info("Application shutdown completed successfully")
                            except asyncio.TimeoutError:
                                logger.warning("Application shutdown timed out, forcing cleanup")
                            except Exception as e:
                                if "This Application is still running" in str(e):
                                    logger.warning("Application is still running during shutdown, forcing cleanup")
                                else:
                                    logger.warning(f"Error during application shutdown: {str(e)}")
                        except Exception as e:
                            logger.warning(f"Error during application shutdown: {str(e)}")
                    
                    # Force cleanup of resources regardless of shutdown success
                    self.application = None
                    self.bot = None
                    
                    # Mark as fully stopped
                    shutdown_successful = True
                    logger.info("Telegram bot stopped successfully")
                    
                except Exception as e:
                    logger.error(f"Error during Telegram bot shutdown: {str(e)}")
                
                # Final force cleanup if shutdown failed
                if not shutdown_successful:
                    logger.warning("Forcing Telegram bot resource cleanup")
                    self.application = None
                    self.bot = None
                
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}")
            # Ensure bot is marked as stopped even if error occurs
            self.is_running = False
            self.application = None
            self.bot = None

    async def send_message(self, message: str, chat_id: Optional[int] = None, parse_mode: str = 'HTML',
                          disable_web_page_preview: bool = True):
        """
        Send a message to a chat.
        
        Args:
            message: Text message to send
            chat_id: Chat ID to send to (defaults to first allowed user if None)
            parse_mode: Message parsing mode (HTML, Markdown, etc.)
            disable_web_page_preview: Whether to disable web previews

        Returns:
            The message sending result
        """
        try:
            # Ensure bot is initialized
            if not self.is_initialized() or not self.application:
                logger.error("Bot not initialized for sending message")
                return False
                
            # If chat_id is None, use the default chat ID
            if chat_id is None and self.allowed_user_ids:
                chat_id = int(self.allowed_user_ids[0])
                
            if not chat_id:
                logger.error("No chat_id provided and no default available")
                return False
                
            # Ensure the application has a bot
            if not hasattr(self.application, 'bot') or not self.application.bot:
                logger.error("Bot not available on application")
                return False
                
            # Send message
            try:
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview
                )
                return True
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}")
            return False

    async def send_trade_update(
        self,
        order_id: Optional[int] = None,
        ticket: Optional[int] = None,  # For backward compatibility
        symbol: str = "",
        action: str = "",
        price: float = 0.0,
        profit: Optional[float] = None,  # For backward compatibility
        pnl: Optional[float] = None,
        r_multiple: Optional[float] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Send a trade update notification.
        
        Args:
            order_id: Trade order ID (preferred)
            ticket: Trade ticket number (backward compatibility)
            symbol: Trading symbol
            action: Trade action (e.g., "OPENED", "CLOSED", "MODIFIED")
            price: Current price
            profit: Trade profit/loss (backward compatibility)
            pnl: Trade profit/loss
            r_multiple: R-multiple value
            reason: Optional reason for the update
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            # Handle backward compatibility
            trade_id = order_id if order_id is not None else ticket
            trade_pnl = pnl if pnl is not None else profit
            
            # Determine action emoji
            if action.upper() == "OPENED":
                action_emoji = "🆕"
                title = "TRADE OPENED"
            elif action.upper() == "CLOSED":
                action_emoji = "🏁"
                title = "TRADE CLOSED"
            elif action.upper() == "MODIFIED":
                action_emoji = "🔄"
                title = "TRADE MODIFIED"
            elif action.upper() == "PARTIAL":
                action_emoji = "⚖️"
                title = "PARTIAL CLOSE"
            else:
                action_emoji = "ℹ️"
                title = "TRADE UPDATE"
                
            # Determine PnL emoji if available
            pnl_emoji = ""
            if trade_pnl is not None:
                if trade_pnl > 0:
                    pnl_emoji = "✅ +"
                elif trade_pnl < 0:
                    pnl_emoji = "❌ "
                else:
                    pnl_emoji = "➖ "
            
            message = f"""{action_emoji} <b>{title}: {symbol}</b>

<b>Details:</b>
• Action: {action}
• Order ID: {trade_id}
• Price: {price:.5f}"""

            if trade_pnl is not None:
                message += f"\n• P/L: {pnl_emoji}{trade_pnl:.2f}"
            
            if r_multiple is not None:
                message += f"\n• R Multiple: {r_multiple:.2f}R"
                
            if reason:
                message += f"\n\n<b>Reason:</b>\n{reason}"
                
            message += f"\n\n<i>{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
            
            return await self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending trade update: {str(e)}")
            return False
    
    async def send_news_alert(
        self,
        symbol: str,
        title: str,
        sentiment: float,
        impact: str,
        source: str
    ):
        """Send news alert to users."""
        try:
            if not self.is_running or self.bot is None:
                logger.error("Cannot send news alert: bot is not initialized")
                # Don't try to initialize here
                return
            
            sentiment_emoji = "🟢" if sentiment > 0 else "🔴" if sentiment < 0 else "⚪"
            
            alert_msg = f"""📰 <b>News Alert</b> 📰

Symbol: {symbol}
Impact: {impact}

Title: {title}
Source: {source}
Sentiment: {sentiment_emoji} {sentiment:.2f}

Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"""
            
            await self.send_message(alert_msg)
        except Exception as e:
            logger.error(f"Error sending news alert: {str(e)}")

    async def send_notification(self, message: str, chat_id: Optional[int] = None, parse_mode: str = 'HTML') -> bool:
        """
        Send a general notification message to all allowed users or a specific user.
        
        Args:
            message: The message to send
            chat_id: Optional specific user to send to
            parse_mode: Message format ('HTML' or 'Markdown')
            
        Returns:
            True if sent successfully to at least one user, False otherwise
        """
        if not self.is_running or not self.bot:
            logger.warning("Telegram bot not running, skipping notification")
            return False
            
        success = False
        
        try:
            if chat_id is not None:
                # Send to specific user if they're allowed
                if str(chat_id) in self.allowed_user_ids:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode=parse_mode,
                        disable_web_page_preview=True
                    )
                    success = True
            else:
                # Send to all allowed users
                for user_id in self.allowed_user_ids:
                    try:
                        await self.bot.send_message(
                            chat_id=int(user_id),
                            text=message,
                            parse_mode=parse_mode,
                            disable_web_page_preview=True
                        )
                        success = True
                    except Exception as e:
                        logger.error(f"Failed to send notification to user {user_id}: {str(e)}")
                        
            return success
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False

    async def register_command_handler(self, command_name: str, handler_function) -> bool:
        """
        Register a custom command handler with the Telegram bot.
        
        Args:
            command_name: The command to register (without leading slash)
            handler_function: Async function to handle the command
                              Should accept args parameter and return response text
        
        Returns:
            Boolean indicating success
        """
        if not self.is_running or not self.application:
            logger.warning(f"Cannot register command /{command_name} - Telegram bot not running")
            return False
            
        try:
            # Create a wrapper function that matches the expected handler signature
            async def command_handler_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
                try:
                    # Store the current update for use in other parts of the code
                    self.last_update = update
                    
                    # Check if user is authorized
                    if not update or not hasattr(update, 'effective_user') or update.effective_user is None:
                        logger.warning(f"Received invalid update for /{command_name} command")
                        return
                    
                    user_id = update.effective_user.id
                    if str(user_id) not in self.allowed_user_ids:
                        if update.message and hasattr(update.message, 'reply_text'):
                            await update.message.reply_text("Unauthorized access.")
                        return
                    
                    # Extract arguments from the message text
                    args = []
                    if update.message and update.message.text:
                        command_parts = update.message.text.split(' ')
                        if len(command_parts) > 1:
                            args = command_parts[1:]
                    
                    # Call the actual handler function with extracted args
                    response = await handler_function(args)
                    
                    # Send the response
                    if response and update.message and hasattr(update.message, 'reply_text'):
                        await update.message.reply_text(
                            response,
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                except Exception as e:
                    error_message = f"Error processing /{command_name} command: {str(e)}"
                    logger.error(error_message)
                    logger.error(traceback.format_exc())
                    if update and update.message and hasattr(update.message, 'reply_text'):
                        await update.message.reply_text(f"Error: {str(e)}")
            
            # Add the command handler to the application
            self.application.add_handler(CommandHandler(command_name, command_handler_wrapper))
            
            # Store in local dictionary for tracking
            self.command_handlers[command_name] = handler_function
            
            logger.info(f"Successfully registered command handler for /{command_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register command handler for /{command_name}: {str(e)}")
            return False 
    
    async def show_history_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Show the date range selection UI for history viewing.
        
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            # Ensure we have a valid update with a chat
            if not update or not update.effective_chat:
                logger.error("Invalid update or missing effective_chat for history selection")
                return
                
            # Create date range options
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            
            # Get today's date
            today = datetime.now().date()
            
            # Create keyboard with date range options
            keyboard = [
                [
                    InlineKeyboardButton("Today", callback_data="history_today"),
                    InlineKeyboardButton("Yesterday", callback_data="history_yesterday")
                ],
                [
                    InlineKeyboardButton("Last 7 days", callback_data="history_7days"),
                    InlineKeyboardButton("Last 30 days", callback_data="history_30days")
                ],
                [
                    InlineKeyboardButton("This month", callback_data="history_this_month"),
                    InlineKeyboardButton("Last month", callback_data="history_last_month")
                ],
                [
                    InlineKeyboardButton("All time", callback_data="history_all")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Check if the application and bot are initialized
            if not self.is_initialized() or not self.application or not self.application.bot:
                logger.error("Bot not initialized for showing history selection")
                if update.message:
                    await update.message.reply_text("❌ Bot is not initialized. Please try again later.")
                return
                
            # We have direct access to the update here, so we can safely use it
            if update.effective_chat:
                await self.application.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="📅 <b>Select a date range for trade history:</b>",
                    reply_markup=reply_markup,
                    parse_mode="HTML"
                )
            else:
                logger.error("Could not show history selection UI: missing chat_id")
                
        except Exception as e:
            logger.error(f"Error showing history selection UI: {str(e)}")
            logger.error(traceback.format_exc()) 

    def set_trading_bot(self, trading_bot):
        """
        Set the trading bot reference after initialization.
        
        Args:
            trading_bot: The trading bot instance
        """
        self.trading_bot = trading_bot
        logger.info("Trading bot reference updated in TelegramBot") 