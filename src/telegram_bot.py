from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from typing import Dict, List, Optional
from loguru import logger
import json
from datetime import datetime, timedelta, UTC
import pandas as pd
import asyncio
from httpx import ConnectError

from config.config import TELEGRAM_CONFIG, TRADING_CONFIG

class TelegramBot:
    def __init__(self):
        self.trading_enabled = False
        self.application = None
        self.bot = None
        self.trade_history = []
        self.is_running = False  # Add state tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        self.allowed_user_ids = TELEGRAM_CONFIG.get("allowed_user_ids", [])
    
    async def initialize(self, config):
        """Initialize the Telegram bot."""
        try:
            logger.info("Starting Telegram bot initialization...")
            self.config = config
            
            # Validate bot token
            if not TELEGRAM_CONFIG.get("bot_token"):
                logger.error("No bot token provided in TELEGRAM_CONFIG")
                return False
            
            # Stop any existing application
            await self.stop()
            
            logger.info("Building Telegram application...")
            self.application = Application.builder().token(TELEGRAM_CONFIG["bot_token"]).build()
            
            # Add command handlers with logging
            logger.info("Registering command handlers...")
            handlers = [
                ("start", self.start_command),
                ("enable", self.handle_enable_command),
                ("disable", self.handle_disable_command),
                ("status", self.status_command),
                ("metrics", self.metrics_command),
                ("history", self.history_command),
                ("help", self.help_command),
                ("stop", self.stop_command)
            ]
            
            for command, handler in handlers:
                self.application.add_handler(CommandHandler(command, handler))
                logger.debug(f"Registered handler for /{command} command")
            
            # Add error handler
            self.application.add_error_handler(self._error_handler)
            
            # Add general message handler for debugging
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._debug_message_handler)
            )
            
            logger.info("Starting Telegram application...")
            
            # Start the bot with retries and exponential backoff
            max_retries = 3
            retry_interval = 5
            
            for attempt in range(max_retries):
                try:
                    await self.application.initialize()
                    await self.application.start()
                    await self.application.updater.start_polling()
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
            
            # Validate allowed user IDs
            if not TELEGRAM_CONFIG.get("allowed_user_ids"):
                logger.warning("No allowed user IDs configured!")
            else:
                logger.info(f"Configured for {len(TELEGRAM_CONFIG['allowed_user_ids'])} authorized users")
            
            # Send startup message to all allowed users
            startup_message = """🤖 <b>Trading Bot Started</b>

Bot is now online and ready to receive commands.
Use /help to see available commands.

⚠️ Send any message to verify connection."""
            
            for user_id in TELEGRAM_CONFIG["allowed_user_ids"]:
                try:
                    await self.bot.send_message(
                        chat_id=user_id,
                        text=startup_message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sent startup message to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send startup message to {user_id}: {str(e)}")
            
            logger.info("Telegram bot initialization completed successfully")
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
                
                if update and hasattr(update, 'effective_chat'):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=error_msg,
                        parse_mode='HTML'
                    )
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
    
    async def _debug_message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle non-command messages for debugging."""
        user_id = update.effective_user.id
        message = update.message.text
        logger.debug(f"Received message from {user_id}: {message}")
        
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            await update.message.reply_text(
                "✅ Message received! Use /help to see available commands.",
                parse_mode='HTML'
            )
        else:
            logger.warning(f"Message from unauthorized user {user_id}: {message}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        await update.message.reply_text(
            "Welcome to the Trading Bot! Use /help to see available commands."
        )
    
    async def enable_trading_core(self):
        """Core functionality to enable trading."""
        try:
            self.trading_enabled = True
            await self.send_message("Trading has been enabled.")
            logger.info("Trading enabled via Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to enable trading: {str(e)}")
            return False

    async def disable_trading_core(self):
        """Core functionality to disable trading."""
        try:
            self.trading_enabled = False
            await self.send_message("Trading has been disabled.")
            logger.info("Trading disabled via Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to disable trading: {str(e)}")
            return False

    async def handle_enable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /enable command to enable trading."""
        try:
            if str(update.effective_user.id) in TELEGRAM_CONFIG["allowed_user_ids"]:
                await self.enable_trading_core()
                await update.message.reply_text("Trading has been enabled.")
            else:
                await update.message.reply_text("Unauthorized access.")
        except Exception as e:
            await update.message.reply_text(f"Failed to enable trading: {str(e)}")

    async def handle_disable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /disable command to disable trading."""
        try:
            if str(update.effective_user.id) in TELEGRAM_CONFIG["allowed_user_ids"]:
                await self.disable_trading_core()
                await update.message.reply_text("Trading has been disabled.")
            else:
                await update.message.reply_text("Unauthorized access.")
        except Exception as e:
            await update.message.reply_text(f"Failed to disable trading: {str(e)}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /status command to check trading status."""
        status = "enabled" if self.trading_enabled else "disabled"
        await update.message.reply_text(f"Trading is currently {status}.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            help_text = """🤖 <b>Trading Bot Commands</b>

🎮 <b>Control Commands</b>
/start - Initialize the bot
/enable - Enable trading
/disable - Disable trading
/stop - Stop bot and cancel trades
/status - Check bot status

📊 <b>Information Commands</b>
/metrics - View performance metrics
/history - View trade history
/help - Show this help message

All commands are restricted to authorized users only."""
            
            await update.message.reply_text(help_text, parse_mode='HTML')
        else:
            await update.message.reply_text("Unauthorized access.")

    async def metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /metrics command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            win_rate = (self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades'] * 100) if self.performance_metrics['total_trades'] > 0 else 0
            
            metrics_text = f"""📊 <b>Performance Metrics</b>

Total Trades: {self.performance_metrics['total_trades']}
Winning Trades: {self.performance_metrics['winning_trades']}
Losing Trades: {self.performance_metrics['losing_trades']}
Win Rate: {win_rate:.2f}%
Total Profit: {self.performance_metrics['total_profit']:.2f}
Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}%

Last Updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"""
            
            await update.message.reply_text(metrics_text, parse_mode='HTML')
        else:
            await update.message.reply_text("Unauthorized access.")

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /history command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            if not self.trade_history:
                await update.message.reply_text("No trade history available.")
                return
            
            # Get last 10 trades
            recent_trades = self.trade_history[-10:]
            history_text = "📜 <b>Recent Trade History</b>\n\n"
            
            for trade in recent_trades:
                history_text += f"""Trade ID: {trade['id']}
Symbol: {trade['symbol']}
Type: {trade['type']}
Entry: {trade['entry']:.5f}
Exit: {trade['exit']:.5f}
PnL: {trade['pnl']:.2f}
Time: {trade['time']}
------------------------\n"""
            
            await update.message.reply_text(history_text, parse_mode='HTML')
        else:
            await update.message.reply_text("Unauthorized access.")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /stop command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            self.trading_enabled = False
            # Signal to close all trades
            await self.send_message("⚠️ Stopping bot and closing all trades...")
            # The actual trade closing logic should be handled by the main trading bot
            await update.message.reply_text("Bot stopped and all trades are being closed.")
        else:
            await update.message.reply_text("Unauthorized access.")
    
    async def send_setup_alert(self, symbol: str, timeframe: str, setup_type: str, confidence: float):
        """Send setup formation alert to users."""
        if self.bot:
            alert_msg = f"""🎯 <b>Setup Alert</b> 🎯

Symbol: {symbol}
Timeframe: {timeframe}
Setup: {setup_type}
Confidence: {confidence:.1f}%

Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"""
            
            await self.send_message(alert_msg)

    async def send_management_alert(self, trade_id: int, symbol: str, action: str, old_value: float, new_value: float, reason: str):
        """Send trade management alert to users."""
        if self.bot:
            alert_msg = f"""📝 <b>Trade Management Alert</b> 📝

Trade ID: {trade_id}
Symbol: {symbol}
Action: {action}
Old Value: {old_value:.5f}
New Value: {new_value:.5f}
Reason: {reason}

Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"""
            
            await self.send_message(alert_msg)

    def update_metrics(self, trade_result: Dict):
        """Update performance metrics with new trade result."""
        self.performance_metrics['total_trades'] += 1
        if trade_result['pnl'] > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        self.performance_metrics['total_profit'] += trade_result['pnl']
        
        # Add trade to history
        self.trade_history.append({
            'id': trade_result['id'],
            'symbol': trade_result['symbol'],
            'type': trade_result['type'],
            'entry': trade_result['entry'],
            'exit': trade_result['exit'],
            'pnl': trade_result['pnl'],
            'time': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Keep only last 100 trades in history
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    async def send_performance_update(
        self,
        chat_id: int,
        total_trades: int,
        winning_trades: int,
        total_profit: float
    ):
        """Send a performance update."""
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
        """Send a trade alert to a specific chat."""
        message = self.format_alert(symbol, direction, entry, sl, tp, confidence, reason)
        await self.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='HTML'
        )

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
        return f"""🔔 <b>Trade Alert</b>

Symbol: {symbol}
Direction: {direction}
Entry: {entry:.5f}
Stop Loss: {sl:.5f}
Take Profit: {tp:.5f}
Confidence: {confidence*100:.0f}%
Reason: {reason}

⚠️ Trade at your own risk."""

    async def notify_error(self, chat_id: int, error: str):
        """Send an error notification."""
        error_message = f"""⚠️ <b>Error Alert</b>

{error}

Please check the logs for more details."""
        await self.bot.send_message(
            chat_id=chat_id,
            text=error_message,
            parse_mode='HTML'
        )

    async def notify_performance(self, chat_id: str, data: Dict):
        """Send performance update to specified chat."""
        try:
            win_rate = (data['winning_trades'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0
            
            message = f"""📊 <b>Performance Update</b> 📊

Total Trades: {data['total_trades']}
Winning Trades: {data['winning_trades']}
Win Rate: {win_rate:.1f}%
Total Profit: {data['profit']:.2f}

Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"""

            if self.bot:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.error(f"Error sending performance update: {str(e)}")

    async def process_command(self, message):
        """Process a command message."""
        if message.text == "/start":
            welcome_message = """👋 <b>Welcome to Trading Bot!</b>

Thank you for using our service. Use /help to see available commands.

Stay profitable! 📈"""
            await self.bot.send_message(
                chat_id=message.chat.id,
                text=welcome_message,
                parse_mode='HTML'
            )

    async def check_auth(self, chat_id: int) -> bool:
        """Check if a user is authorized."""
        return str(chat_id) in self.allowed_user_ids
    
    async def send_error_alert(self, message: str) -> bool:
        """Send error alert to Telegram with retry logic and timeout handling."""
        if not self.is_running or not self.bot:
            logger.warning("Telegram bot not running, skipping error alert")
            return False

        max_retries = 3
        retry_delay = 2  # seconds
        success = False
        
        for user_id in self.allowed_user_ids:
            for attempt in range(max_retries):
                try:
                    async with asyncio.timeout(5):  # 5 second timeout
                        await self.bot.send_message(
                            chat_id=int(user_id),  # Convert string ID to int
                            text=f"🚨 Error Alert:\n{message}",
                            parse_mode='HTML'
                        )
                        success = True
                        break
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Telegram alert timed out, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to send error alert to {user_id} after {max_retries} attempts: Timed out")
                except Exception as e:
                    logger.error(f"Failed to send error alert to {user_id}: {str(e)}")
                    break
                    
        return success
    
    async def stop(self):
        """Stop the Telegram bot."""
        try:
            if self.is_running:
                logger.info("Stopping Telegram bot...")
                try:
                    # First disable trading
                    self.trading_enabled = False
                    
                    # Get the current event loop
                    loop = asyncio.get_running_loop()
                    
                    # Stop the updater first if it exists and is running
                    if hasattr(self, 'application') and self.application:
                        if hasattr(self.application, 'updater') and self.application.updater:
                            try:
                                # Cancel polling task if it exists
                                if hasattr(self.application.updater, '_polling_task'):
                                    self.application.updater._polling_task.cancel()
                                await self.application.updater.stop()
                            except Exception as e:
                                logger.warning(f"Error stopping updater: {str(e)}")
                        
                        # Then shutdown the application
                        try:
                            await self.application.shutdown()
                        except Exception as e:
                            logger.warning(f"Error during application shutdown: {str(e)}")
                        
                        # Clear application reference
                        self.application = None
                    
                    # Clear bot reference
                    self.bot = None
                    
                    # Update state
                    self.is_running = False
                    
                    logger.info("Telegram bot stopped successfully")
                    
                except Exception as e:
                    logger.error(f"Error during Telegram bot shutdown: {str(e)}")
                    # Ensure states are cleared even if there were errors
                    self.is_running = False
                    self.trading_enabled = False
                    self.application = None
                    self.bot = None
                
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}")
            raise
    
    async def send_message(self, message: str):
        """Send message to all allowed users."""
        if not self.application:
            logger.error("Telegram bot not initialized")
            return
        
        for user_id in TELEGRAM_CONFIG["allowed_user_ids"]:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {str(e)}")
    
    async def send_trade_update(
        self,
        trade_id: int,
        symbol: str,
        action: str,
        price: float,
        pnl: Optional[float] = None,
        r_multiple: Optional[float] = None
    ):
        """Send trade update to users."""
        update_msg = f"""📊 <b>Trade Update</b>

Trade ID: {trade_id}
Symbol: {symbol}
Action: {action}
Price: {price:.5f}"""
        
        if pnl is not None:
            update_msg += f"\nPnL: {pnl:.2f}"
        
        if r_multiple is not None:
            update_msg += f"\nR-Multiple: {r_multiple:.2f}"
        
        update_msg += f"\nTime: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(update_msg)
    
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
            pass 