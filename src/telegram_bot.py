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
                
            logger.info("Building Telegram application...")
            self.application = Application.builder().token(TELEGRAM_CONFIG["bot_token"]).build()
            
            # Add command handlers with logging
            logger.info("Registering command handlers...")
            handlers = [
                ("start", self.start_command),
                ("enable", self.enable_trading),
                ("disable", self.disable_trading),
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
            
            # Start the bot with retries
            max_retries = 3
            retry_interval = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    await self.application.initialize()
                    await self.application.start()
                    await self.application.updater.start_polling()
                    break
                except ConnectError as e:
                    logger.error(f"Failed to connect to Telegram API (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_interval} seconds...")
                        await asyncio.sleep(retry_interval)
                        retry_interval *= 2  # Exponential backoff
                    else:
                        logger.error("Failed to connect to Telegram API after multiple retries")
                        await self.send_error_alert("Failed to connect to Telegram API. Bot is shutting down.")
                        return False
            
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
            if self.application:
                try:
                    await self.application.stop()
                except Exception:
                    pass
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
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            welcome_text = """🤖 <b>Trading Bot Initialized</b>

Welcome to your Automated Trading Assistant! The bot is now ready to help you manage your trades.

🔑 <b>Quick Start Guide</b>:
1. Use /enable to start automated trading
2. Use /disable to pause trading
3. Use /metrics to view performance
4. Use /history to see past trades
5. Use /status to check bot status
6. Use /help for full command list

⚠️ Current Status: Trading is {status}

Use /help to see all available commands and features."""

            status = "enabled" if self.trading_enabled else "disabled"
            await update.message.reply_text(
                welcome_text.format(status=status),
                parse_mode='HTML'
            )
            
            # Send initial status update
            metrics_text = f"""📊 <b>Current Statistics</b>

Total Trades: {self.performance_metrics['total_trades']}
Winning Trades: {self.performance_metrics['winning_trades']}
Total Profit: {self.performance_metrics['total_profit']:.2f}

Use /metrics for detailed performance stats."""
            
            await update.message.reply_text(metrics_text, parse_mode='HTML')
        else:
            await update.message.reply_text(
                "⚠️ Unauthorized access. This incident will be logged.",
                parse_mode='HTML'
            )
            logger.warning(f"Unauthorized access attempt from user ID: {user_id}")
    
    async def enable_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enable trading."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            if self.trading_enabled:
                await update.message.reply_text("Trading is already enabled.")
                return
                
            self.trading_enabled = True
            status_msg = """✅ <b>Trading Enabled</b>

The bot will now:
• Analyze markets
• Look for trading setups
• Execute trades when conditions are met
• Send real-time alerts

Use /status to check bot status
Use /disable to stop trading"""
            
            await update.message.reply_text(status_msg, parse_mode='HTML')
            logger.info("Trading enabled via Telegram command")
        else:
            await update.message.reply_text("Unauthorized access.")
    
    async def disable_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Disable trading."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            if not self.trading_enabled:
                await update.message.reply_text("Trading is already disabled.")
                return
                
            self.trading_enabled = False
            status_msg = """🛑 <b>Trading Disabled</b>

• No new trades will be opened
• Existing trades will be monitored
• Alerts will continue

Use /status to check bot status
Use /enable to resume trading"""
            
            await update.message.reply_text(status_msg, parse_mode='HTML')
            logger.info("Trading disabled via Telegram command")
        else:
            await update.message.reply_text("Unauthorized access.")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check bot status."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            status = "enabled" if self.trading_enabled else "disabled"
            
            status_msg = f"""📊 <b>Bot Status Report</b>

Trading Status: {'🟢 Enabled' if self.trading_enabled else '🔴 Disabled'}
Total Trades: {self.performance_metrics['total_trades']}
Active Since: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}

Use /metrics for detailed performance stats
Use /{'disable' if self.trading_enabled else 'enable'} to {'stop' if self.trading_enabled else 'start'} trading"""
            
            await update.message.reply_text(status_msg, parse_mode='HTML')
        else:
            await update.message.reply_text("Unauthorized access.")

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
    
    async def send_error_alert(self, error_message):
        """Send error alert to Telegram."""
        if self.bot:
            message = f"⚠️ ERROR ALERT ⚠️\n\n{error_message}"
            
            for user_id in TELEGRAM_CONFIG["allowed_user_ids"]:
                try:
                    await self.bot.send_message(chat_id=user_id, text=message)
                except Exception as e:
                    logger.error(f"Failed to send error alert to {user_id}: {str(e)}")
        else:
            logger.error("Cannot send error alert. Bot is not initialized.")
    
    async def stop(self):
        """Stop the Telegram bot."""
        if self.application:
            logger.info("Stopping Telegram bot...")
            try:
                if self.application.updater:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("Telegram bot stopped successfully")
            except asyncio.CancelledError:
                logger.warning("Polling task cancelled. This is expected during bot shutdown.")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {str(e)}")
    
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
        pnl: Optional[float] = None
    ):
        """Send trade update to users."""
        update_msg = f"""📊 <b>Trade Update</b> 📊

Trade ID: {trade_id}
Symbol: {symbol}
Action: {action}
Price: {price:.5f}"""
        
        if pnl is not None:
            update_msg += f"\nPnL: {pnl:.2f}"
        
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