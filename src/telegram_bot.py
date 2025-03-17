from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from typing import Dict, List, Optional
from loguru import logger
import json
from datetime import datetime, timedelta, UTC
import pandas as pd
import asyncio
from httpx import ConnectError
import telegram
import traceback

from config.config import TELEGRAM_CONFIG

class TelegramBot:
    def __init__(self):
        self.trading_enabled = False
        self.application = None
        self.bot = None
        self.trade_history = []
        self.is_running = False  # Add state tracking
        self.start_time = datetime.now(UTC)  # Track bot start time
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        self.allowed_user_ids = TELEGRAM_CONFIG.get("allowed_user_ids", [])
        self.command_handlers = {}
    
    async def initialize(self, config):
        """Initialize the Telegram bot."""
        try:
            logger.info("Starting Telegram bot initialization...")
            self.config = config
            self.start_time = datetime.now(UTC)  # Reset start time on initialization
            
            # Validate bot token
            if not TELEGRAM_CONFIG.get("bot_token"):
                logger.error("No bot token provided in TELEGRAM_CONFIG")
                return False
            
            # Validate user IDs
            if not TELEGRAM_CONFIG.get("allowed_user_ids"):
                logger.error("No allowed user IDs configured!")
                return False
            
            # Convert and validate user IDs
            self.allowed_user_ids = [str(uid) for uid in TELEGRAM_CONFIG["allowed_user_ids"]]
            logger.info(f"Configured for {len(self.allowed_user_ids)} users: {self.allowed_user_ids}")
            
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
            
            # Test message to verify each user ID
            startup_message = """🤖 <b>Trading Bot Started</b>

Bot is now online and ready to receive commands.
Use /help to see available commands.

⚠️ Send /start to verify connection."""
            
            successful_users = []
            for user_id in self.allowed_user_ids:
                try:
                    await self.bot.send_message(
                        chat_id=int(user_id),
                        text=startup_message,
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    )
                    successful_users.append(user_id)
                    logger.info(f"Successfully sent startup message to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send startup message to {user_id}: {str(e)}")
            
            if not successful_users:
                logger.error("Could not send messages to any configured users!")
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
        """Handle the /status command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
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
                
                await update.message.reply_text(status_text, parse_mode='HTML')
            except Exception as e:
                logger.error(f"Error generating status: {str(e)}")
                await update.message.reply_text(f"Error retrieving status: {str(e)[:100]}")
        else:
            await update.message.reply_text("Unauthorized access.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
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
            
            await update.message.reply_text(help_text, parse_mode='HTML')
        else:
            await update.message.reply_text("Unauthorized access.")

    async def metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /metrics command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            try:
                # Calculate performance stats with safety checks
                total_trades = self.performance_metrics.get('total_trades', 0)
                
                # Check if there are any trades
                if total_trades == 0:
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
                
                await update.message.reply_text(metrics_text, parse_mode='HTML')
            
            except Exception as e:
                logger.error(f"Error generating metrics: {str(e)}")
                logger.error(traceback.format_exc())
                await update.message.reply_text(
                    f"⚠️ <b>Error retrieving metrics</b>\n\nPlease try again later. Error: {str(e)[:100]}...", 
                    parse_mode='HTML'
                )
        else:
            await update.message.reply_text("Unauthorized access.")

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /history command."""
        user_id = update.effective_user.id
        if str(user_id) in TELEGRAM_CONFIG["allowed_user_ids"]:
            try:
                if not self.trade_history:
                    await update.message.reply_text(
                        "📜 <b>No trade history available yet</b>\n\nHistory will be displayed after completed trades.", 
                        parse_mode='HTML'
                    )
                    return
                
                # Get last 10 trades
                recent_trades = self.trade_history[-10:]
                
                # Validate trade data
                valid_trades = []
                for trade in recent_trades:
                    if 'pnl' in trade and 'symbol' in trade:
                        valid_trades.append(trade)
                    else:
                        logger.warning(f"Skipping invalid trade record: {trade}")
                
                if not valid_trades:
                    await update.message.reply_text(
                        "⚠️ <b>Trade history contains invalid data</b>\n\nPlease contact the bot administrator.", 
                        parse_mode='HTML'
                    )
                    return
                
                # Calculate summary stats
                winning_count = sum(1 for trade in valid_trades if trade.get('pnl', 0) > 0)
                losing_count = sum(1 for trade in valid_trades if trade.get('pnl', 0) <= 0)
                total_pnl = sum(trade.get('pnl', 0) for trade in valid_trades)
                
                # Create summary header
                history_text = f"""📜 <b>RECENT TRADE HISTORY</b> 📜
<i>Last {len(valid_trades)} trades | {winning_count} wins, {losing_count} losses | Total: {total_pnl:.2f}</i>

"""
                
                # Add each trade with emoji indicators
                for i, trade in enumerate(valid_trades, 1):
                    # Determine symbols based on trade result
                    result_emoji = "✅" if trade.get('pnl', 0) > 0 else "❌" if trade.get('pnl', 0) < 0 else "➖"
                    
                    # Determine type emoji
                    trade_type = trade.get('type', 'Unknown')
                    if isinstance(trade_type, str) and trade_type.lower() in ['buy', 'long']:
                        type_emoji = "🟢"
                    elif isinstance(trade_type, str) and trade_type.lower() in ['sell', 'short']:
                        type_emoji = "🔴"
                    else:
                        type_emoji = "⚪"
                    
                    # Get trade values with defaults for missing data
                    entry = trade.get('entry', 0.0)
                    exit = trade.get('exit', 0.0)
                    pnl = trade.get('pnl', 0.0)
                    symbol = trade.get('symbol', 'Unknown')
                    trade_time = trade.get('time', 'Unknown time')
                    
                    # Calculate profit percentage if we have both entry and exit
                    profit_pct = ""
                    if entry != 0:
                        pct = (exit - entry) / entry * 100
                        pct_sign = "+" if pct > 0 else ""
                        profit_pct = f" ({pct_sign}{pct:.2f}%)"
                    
                    history_text += f"""<b>{i}. {result_emoji} {symbol}</b> {type_emoji} {trade_type}
   Entry: <b>{entry:.5f}</b> → Exit: <b>{exit:.5f}</b>
   P/L: <b>{pnl:.2f}</b>{profit_pct}
   <i>{trade_time}</i>
   
"""
                
                await update.message.reply_text(history_text, parse_mode='HTML')
            
            except Exception as e:
                logger.error(f"Error retrieving trade history: {str(e)}")
                logger.error(traceback.format_exc())
                await update.message.reply_text(
                    f"⚠️ <b>Error retrieving trade history</b>\n\nPlease try again later. Error: {str(e)[:100]}...", 
                    parse_mode='HTML'
                )
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

    async def send_management_alert(self, message: str, alert_type: str = "info") -> None:
        """Send trade management alert message."""
        try:
            # Check if bot is running
            if not self.is_running or not self.bot:
                logger.warning("Cannot send management alert - Telegram bot not running")
                return
                
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
            if not hasattr(self, 'chat_id') or not self.chat_id:
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
            else:
                # Send to default chat
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to send management alert: {str(e)}")
            logger.error(traceback.format_exc())
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
                'exit': trade_result.get('exit', 0.0),
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
        additional_info: dict = None
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

        if additional_info:
            error_msg += "\n\nAdditional Information:"
            for key, value in additional_info.items():
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
                                disable_web_page_preview=True,
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
        """Send a trade alert to a specific chat with improved error handling."""
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return

        message = self.format_alert(symbol, direction, entry, sl, tp, confidence, reason)
        max_retries = 5
        base_delay = 1  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Increase timeout for the request
                async with asyncio.timeout(15):  # 15 second timeout
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML',
                        disable_web_page_preview=True,  # Speed up sending
                        disable_notification=False  # Enable notifications for trade alerts
                    )
                logger.info(f"Successfully sent trade alert for {symbol}")
                return
                
            except asyncio.TimeoutError:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Timeout sending trade alert (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    
            except telegram.error.RetryAfter as e:
                # Handle rate limiting
                logger.warning(f"Rate limited, waiting {e.retry_after} seconds")
                await asyncio.sleep(e.retry_after)
                continue
                
            except telegram.error.TimedOut:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Telegram timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    
            except telegram.error.NetworkError as e:
                delay = base_delay * (2 ** attempt)
                logger.error(f"Network error: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error sending trade alert: {str(e)}")
                if "Bad Request" in str(e):
                    # Don't retry on bad requests
                    break
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    
        logger.error(f"Failed to send trade alert for {symbol} after {max_retries} attempts")
        # Try to send error notification through alternative method
        await self.send_error_alert(f"Failed to send trade alert for {symbol} due to connection issues")

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
            await self.bot.send_message(
                chat_id=chat_id,
                text=error_message,
                parse_mode='HTML',
                disable_notification=False  # Important errors should trigger notifications
            )
            logger.info(f"Sent error notification to user {chat_id}")
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

        max_retries = 5
        retry_delay = 1  # seconds
        success = False
        
        for user_id in self.allowed_user_ids:
            for attempt in range(max_retries):
                try:
                    async with asyncio.timeout(10):  # 10 second timeout
                        await self.bot.send_message(
                            chat_id=int(user_id),
                            text=f"🚨 Error Alert:\n{message}",
                            parse_mode='HTML',
                            disable_web_page_preview=True,  # Speed up message sending
                            disable_notification=True  # Don't notify for errors
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
                except ConnectionError as e:
                    logger.error(f"Connection error sending to {user_id}: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                except Exception as e:
                    logger.error(f"Error sending alert to {user_id}: {str(e)}")
                    if "Too Many Requests" in str(e):
                        await asyncio.sleep(5)  # Wait longer for rate limits
                        continue
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    
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
    
    async def send_message(self, message: str, chat_id: Optional[int] = None, parse_mode: str = 'HTML',
                          disable_web_page_preview: bool = True):
        """Send message to specific user or all allowed users."""
        try:
            if not self.is_running:
                return
                
            if chat_id:
                # Send to specific user if they're allowed
                if str(chat_id) in self.allowed_user_ids:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode=parse_mode,
                        disable_web_page_preview=disable_web_page_preview
                    )
            else:
                # Send to all allowed users
                for user_id in self.allowed_user_ids:
                    try:
                        await self.bot.send_message(
                            chat_id=int(user_id),
                            text=message,
                            parse_mode=parse_mode,
                            disable_web_page_preview=disable_web_page_preview
                        )
                    except Exception as e:
                        logger.error(f"Failed to send message to user {user_id}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

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

    async def send_notification(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a general notification message to all allowed users.
        
        Args:
            message: The message to send
            parse_mode: Message format ('HTML' or 'Markdown')
            
        Returns:
            True if sent successfully to at least one user, False otherwise
        """
        if not self.is_running or not self.bot:
            logger.warning("Telegram bot not running, skipping notification")
            return False
            
        success = False
        
        for user_id in self.allowed_user_ids:
            try:
                await self.bot.send_message(
                    chat_id=int(user_id),
                    text=message,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
                success = True
                logger.debug(f"Notification sent to user {user_id}")
            except Exception as e:
                logger.error(f"Failed to send notification to user {user_id}: {str(e)}")
                
        return success 

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
                    # Check if user is authorized
                    user_id = update.effective_user.id
                    if str(user_id) not in self.allowed_user_ids:
                        await update.message.reply_text("Unauthorized access.")
                        return
                    
                    # Extract arguments from the message
                    args = context.args if hasattr(context, 'args') else []
                    
                    # Call the actual handler function
                    response = await handler_function(args)
                    
                    # Send the response
                    if response:
                        await update.message.reply_text(
                            response,
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                except Exception as e:
                    error_message = f"Error processing /{command_name} command: {str(e)}"
                    logger.error(error_message)
                    logger.error(traceback.format_exc())
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