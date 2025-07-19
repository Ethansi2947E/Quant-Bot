from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, TYPE_CHECKING, Optional, Callable, Union, Awaitable
import traceback

# Use TYPE_CHECKING to avoid runtime imports
if TYPE_CHECKING:
    from src.trading_bot import TradingBot
    from src.telegram.telegram_bot import TelegramBot
    from src.mt5_handler import MT5Handler

from loguru import logger

class TelegramCommandHandler:
    """
    Handler for all Telegram bot commands, extracted from TradingBot for modularity.
    
    This class handles all Telegram command processing, delegating actual business
    logic operations to the appropriate components.
    """
    
    def __init__(self, trading_bot: "TradingBot", mt5_handler: "MT5Handler"):
        """
        Initialize the TelegramCommandHandler.
        
        Args:
            trading_bot: Reference to the trading bot instance for executing actions
            mt5_handler: Reference to the MT5 handler for direct data access
        """
        self.trading_bot = trading_bot
        self.mt5_handler = mt5_handler
        self.command_handlers = {}
        
    def set_trading_bot(self, trading_bot: "TradingBot"):
        """
        Set the trading bot reference after initialization.
        
        Args:
            trading_bot: The trading bot instance
        """
        self.trading_bot = trading_bot
    
    def register_command(self, command: str, handler: Callable[[List[str]], Union[str, Awaitable[str]]]) -> None:
        """
        Register a command handler directly with the TelegramCommandHandler.
        
        Args:
            command: Command name without slash prefix (e.g., "status")
            handler: Function to handle the command, should accept args list
        """
        logger.debug(f"Registering command handler for /{command}")
        self.command_handlers[command.lower()] = handler
        
    async def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        """
        Handle a command directly using registered handlers.
        
        Args:
            command: Command name without slash prefix
            args: Command arguments
            
        Returns:
            Optional response message
        """
        command = command.lower()
        
        if command in self.command_handlers:
            try:
                handler = self.command_handlers[command]
                result = handler(args)
                
                # Handle both async and sync handlers
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                logger.error(f"Error handling command /{command}: {str(e)}")
                logger.error(traceback.format_exc())
                return f"Error handling command /{command}: {str(e)}"
        
        return None
        
    async def register_all_commands(self, telegram_bot: "TelegramBot") -> None:
        """
        Register all command handlers with the Telegram bot.
        
        Args:
            telegram_bot: The TelegramBot instance to register commands with
        """
        if not telegram_bot or not hasattr(telegram_bot, 'is_running') or not telegram_bot.is_running:
            logger.warning("Telegram bot not running, skipping command registration")
            return
            
        try:
            # First, set up the command menu if it's supported
            if hasattr(telegram_bot, 'application'):
                from telegram.ext import CallbackQueryHandler
                
                # Add callback query handler for inline buttons
                if telegram_bot.application:
                    telegram_bot.application.add_handler(
                        CallbackQueryHandler(self.handle_callback_query)
                    )
                
                logger.info("Registered callback query handler for interactive buttons")
                
                # Show the command keyboard to users in the allowed list
                # This will happen when the bot starts if the keyboard_shown flag is False
                if not telegram_bot.keyboard_shown and hasattr(telegram_bot, 'allowed_user_ids'):
                    try:
                        # Only try to show menu if we have active users
                        if telegram_bot.allowed_user_ids:
                            # We don't have direct access to chat_id, so we'll use the menu command
                            # to show the keyboard the next time the user interacts with the bot
                            logger.info("Command keyboard will be shown on next user interaction")
                    except Exception as e:
                        logger.error(f"Error preparing command keyboard: {str(e)}")
            
            # First, register all direct command handlers
            for command, handler in self.command_handlers.items():
                await telegram_bot.register_command_handler(command, handler)
                logger.debug(f"Registered command handler for /{command}")
            
            # Then register additional commands if they're not already registered
            # Status and information commands
            if "status" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "status", 
                    self.trading_bot.handle_status_command
                )
            
            # Add metrics and history commands
            if "metrics" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "metrics",
                    self.handle_metrics_command
                )
            
            if "history" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "history",
                    self.handle_history_command
                )
            
            # Register additional commands for keyboard menu buttons
            if "daily" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "daily",
                    self.handle_daily_command
                )
            
            if "balance" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "balance",
                    self.handle_balance_command
                )
            
            if "statustable" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "statustable",
                    self.handle_status_table_command
                )
            
            if "performance" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "performance",
                    self.handle_performance_command
                )
            
            if "count" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "count",
                    self.handle_count_command
                )
            
            if "listsignalgenerators" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "listsignalgenerators", 
                    self.trading_bot.handle_list_signal_generators_command
                )
            
            if "setsignalgenerator" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "setsignalgenerator", 
                    self.trading_bot.handle_set_signal_generator_command
                )
            
            # Trading control commands
            if "enable" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "enable", 
                    lambda args: self.trading_bot.enable_trading()
                )
            
            if "disable" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "disable", 
                    lambda args: self.trading_bot.disable_trading()
                )
            
            # Risk management commands
            if "enabletrailingstop" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "enabletrailingstop",
                    self.trading_bot.handle_enable_trailing_stop_command
                )
            
            if "disabletrailingstop" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "disabletrailingstop",
                    self.trading_bot.handle_disable_trailing_stop_command
                )
            
            # Position management commands
            if hasattr(self.trading_bot, 'handle_enable_position_additions_command') and "enablepositionadditions" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "enablepositionadditions",
                    self.trading_bot.handle_enable_position_additions_command
                )
            
            if hasattr(self.trading_bot, 'handle_disable_position_additions_command') and "disablepositionadditions" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "disablepositionadditions",
                    self.trading_bot.handle_disable_position_additions_command
                )
            
            # Shutdown commands
            if "enablecloseonshutdown" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "enablecloseonshutdown", 
                    self.trading_bot.handle_enable_close_on_shutdown_command
                )
            
            if "disablecloseonshutdown" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "disablecloseonshutdown", 
                    self.trading_bot.handle_disable_close_on_shutdown_command
                )
            
            if "shutdown" not in self.command_handlers:
                await telegram_bot.register_command_handler(
                    "shutdown",
                    self.trading_bot.handle_shutdown_command
                )
            
            
            logger.info("Successfully registered telegram commands")
            
        except Exception as e:
            logger.error(f"Error registering commands: {str(e)}")
            logger.error(traceback.format_exc())
            
    async def handle_callback_query(self, update, context):
        """
        Handle callback queries from inline keyboard buttons.
        
        Args:
            update: The update containing the callback query
            context: The context for the callback query
        """
        try:
            query = update.callback_query
            
            # Always answer the callback query to remove the loading indicator
            await query.answer()
            
            # Get the data from the callback
            data = query.data
            logger.debug(f"Received callback query with data: {data}")
            
            # Handle history command callbacks
            if data.startswith('history:'):
                await self.handle_history_callback(update, context)
            # Handle other callback types as they're implemented
            else:
                await query.edit_message_text(
                    f"Unhandled callback type: {data}",
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.error(f"Error handling callback query: {str(e)}")
            logger.error(traceback.format_exc())
            
    async def handle_history_callback(self, update, context):
        """
        Handle callback queries specifically for history command.
        
        Args:
            update: The update containing the callback query
            context: The context for the callback query
        """
        try:
            query = update.callback_query
            data = query.data
            
            # Parse the callback data
            parts = data.split(':')
            
            if len(parts) >= 2:
                if parts[1] == 'custom':
                    # Show date picker or prompt for custom date range
                    await query.edit_message_text(
                        "📆 <b>Custom Date Range</b>\n\n"
                        "Please use the following format to specify a date range:\n\n"
                        "<code>/history from=YYYY-MM-DD to=YYYY-MM-DD</code>\n\n"
                        "For example:\n"
                        "<code>/history from=2023-01-01 to=2023-01-31</code>",
                        parse_mode='HTML'
                    )
                elif parts[1].startswith('days='):
                    try:
                        # Extract the number of days
                        days = int(parts[1].split('=')[1])
                        
                        # Check if CSV export is requested
                        export_csv = False
                        if len(parts) > 2 and parts[2] == 'csv':
                            export_csv = True
                        
                        # For CSV exports, show a processing message first
                        if export_csv:
                            try:
                                await query.edit_message_text(
                                    "⏳ <b>Generating Trade History Export...</b>\n\n"
                                    f"Preparing data for the last {days} days.\n"
                                    "This may take a moment for large datasets.\n\n"
                                    "<i>Files will be sent to this chat when ready.</i>",
                                    parse_mode='HTML'
                                )
                            except Exception as msg_e:
                                logger.warning(f"Could not update message during CSV preparation: {str(msg_e)}")
                        
                        # Build args list for the history command
                        args = [parts[1]]
                        if export_csv:
                            args.append('csv')
                            
                        # Call the history command with the parsed arguments
                        try:
                            response = await self.handle_history_command(args)
                            
                            # If there's a response and it's not a CSV export, edit the message
                            if response and not export_csv:
                                await query.edit_message_text(
                                    response,
                                    parse_mode='HTML'
                                )
                            # If there's a response for CSV export, update the message
                            elif response and export_csv:
                                try:
                                    await query.edit_message_text(
                                        response,
                                        parse_mode='HTML'
                                    )
                                except Exception as edit_e:
                                    logger.warning(f"Could not update export status message: {str(edit_e)}")
                            # If no response and CSV was requested, message was likely already sent by the command
                            elif not response and export_csv:
                                logger.info("CSV export processing, files will be sent directly")
                                # The files are being sent separately, no need to update this message
                            else:
                                # If no response for some other reason, show an error
                                await query.edit_message_text(
                                    "⚠️ <b>Error processing request</b>\n\n"
                                    "Unable to generate history. Please try again or use the command directly.",
                                    parse_mode='HTML'
                                )
                        except Exception as cmd_e:
                            logger.error(f"Error in handle_history_command: {str(cmd_e)}")
                            logger.error(traceback.format_exc())
                            
                            # Show error message to user
                            await query.edit_message_text(
                                f"⚠️ <b>Error processing trade history</b>\n\n"
                                f"An error occurred: {str(cmd_e)}\n\n"
                                "Please try again or use the command directly with:\n"
                                f"<code>/history {parts[1]} {('csv' if export_csv else '')}</code>",
                                parse_mode='HTML'
                            )
                            
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing days value: {str(e)}")
                        await query.edit_message_text(
                            f"⚠️ <b>Error</b>: Invalid days format in '{parts[1]}'.",
                            parse_mode='HTML'
                        )
            else:
                logger.warning(f"Invalid history callback data format: {data}")
                await query.edit_message_text(
                    "⚠️ <b>Error</b>: Invalid callback data format.",
                    parse_mode='HTML'
                )
                
        except Exception as e:
            logger.error(f"Error handling history callback: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to provide some feedback to the user
            try:
                await query.edit_message_text(
                    "⚠️ <b>Unexpected error occurred</b>\n\n"
                    "Could not process your request. Please try again or contact support.",
                    parse_mode='HTML'
                )
            except:
                # If we can't even update the message, just pass
                pass
            
    async def handle_daily_command(self, args: List[str]) -> str:
        """
        Handler for the /daily command.
        Shows trading history for the current day.
        
        Args:
            args: List of command arguments
            
        Returns:
            Daily trading history summary
        """
        try:
            # Get today's date range
            today = datetime.now()
            start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Format dates for display
            date_str = today.strftime("%Y-%m-%d")
            
            # Check if trading bot and necessary components are initialized
            if not hasattr(self.trading_bot, 'mt5_handler') or self.trading_bot.mt5_handler is None:
                logger.error("MT5 handler not available")
                return f"⚠️ <b>Daily Report ({date_str})</b>\n\nMT5 connection not available. Cannot retrieve trading history."
            
            # Get history using date range
            history_text = await self.handle_history_command([start_date.strftime("%Y-%m-%d"), 
                                                            end_date.strftime("%Y-%m-%d")])
            
            # If no trades were found, return a specific message
            if "No trades found" in history_text:
                return f"📅 <b>Daily Report ({date_str})</b>\n\nNo trades found for today."
            
            # Otherwise, prepend with daily header
            return f"📅 <b>Daily Report ({date_str})</b>\n\n{history_text}"
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            logger.error(traceback.format_exc())
            return f"⚠️ <b>Error retrieving daily report</b>\n\nPlease try again later. Error: {str(e)[:100]}..."
    
    async def handle_balance_command(self, args: List[str]) -> str:
        """
        Handler for the /balance command to show account balance.
        
        Args:
            args: Command arguments (unused for this command)
            
        Returns:
            Formatted balance information
        """
        try:
            # Get MT5 account balance
            from src.mt5_handler import MT5Handler
            mt5_handler = MT5Handler.get_instance()
            
            if mt5_handler and mt5_handler.connected:
                # Get account info
                account_info = mt5_handler.get_account_info()
                
                if account_info:
                    # Extract relevant information
                    balance = account_info.get('balance', 0)
                    equity = account_info.get('equity', 0)
                    profit = account_info.get('profit', 0)
                    margin = account_info.get('margin', 0)
                    free_margin = account_info.get('margin_free', 0)
                    margin_level = account_info.get('margin_level', 0)
                    
                    # Create readable output
                    balance_text = f"""💰 <b>ACCOUNT BALANCE</b> 💰

<b>💵 Balance:</b> {balance:.2f}
<b>💎 Equity:</b> {equity:.2f}
<b>📊 Current P/L:</b> {"+" if profit > 0 else ""}{profit:.2f}

<b>⚖️ Margin:</b>
• Used: {margin:.2f}
• Free: {free_margin:.2f}
• Level: {margin_level:.2f}%

<i>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"""
                    
                    return balance_text
                else:
                    return "⚠️ <b>Unable to retrieve account information</b>\n\nMT5 connection exists but account data is unavailable."
            else:
                return "⚠️ <b>MT5 Not Connected</b>\n\nCannot retrieve account balance without an active MT5 connection."
                
        except Exception as e:
            logger.error(f"Error retrieving account balance: {str(e)}")
            logger.error(traceback.format_exc())
            return f"⚠️ <b>Error retrieving account balance</b>\n\nPlease try again later. Error: {str(e)[:100]}..."
    
    async def handle_status_table_command(self, args: List[str]) -> str:
        """
        Handle the /statustable command to display open positions in a tabular format.
        
        Args:
            args: Command arguments (unused for this command)
            
        Returns:
            Formatted status table or error message
        """
        # Access MT5Handler through the trading bot instead of directly
        if not hasattr(self.trading_bot, 'mt5_handler') or not self.trading_bot.mt5_handler:
            logger.error("MT5Handler not available through trading_bot")
            return "⚠️ Error: MT5 handler not available. Cannot retrieve positions."
            
        positions = self.trading_bot.mt5_handler.get_open_positions()
        
        if not positions:
            # Get connection status to provide better feedback
            mt5_connected = self.trading_bot.mt5_handler.connected if hasattr(self.trading_bot.mt5_handler, 'connected') else False
            if not mt5_connected:
                return "⚠️ MT5 not connected. Please check your connection and try again."
            return "📋 No open positions currently."
        
        # Create a table header
        header = "<pre>| Symbol    | Type | Lots  | Entry    | Current  | P/L   |</pre>"
        table_rows = []
        
        for pos in positions:
            symbol = pos.get('symbol', 'Unknown')
            pos_type = pos.get('type', 0)  # Default to 0 (buy) if not specified
            volume = pos.get('volume', 0.0)
            entry_price = pos.get('price_open', 0.0)
            current_price = pos.get('price_current', 0.0)
            profit = pos.get('profit', 0.0)
            
            # Format the type (0 is buy, 1 is sell)
            type_str = "BUY" if pos_type == 0 else "SELL"
            
            # Format row with fixed widths
            row = f"<pre>| {symbol:<9} | {type_str:<4} | {volume:.2f} | {entry_price:<8.5f} | {current_price:<8.5f} | {profit:+.2f} |</pre>"
            table_rows.append(row)
        
        # Combine all parts
        result = "📋 <b>OPEN POSITIONS</b>\n\n" + header + "\n" + "\n".join(table_rows)
        
        # Add totals if there are multiple positions
        if len(positions) > 1:
            total_profit = sum(pos.get('profit', 0.0) for pos in positions)
            result += f"\n\n<b>Total P/L:</b> {total_profit:+.2f}"
        
        return result
    
    async def handle_performance_command(self, args: List[str]) -> str:
        """
        Handler for the /performance command.
        Shows performance summary calculated directly from MT5 trade history.
        
        Args:
            args: Command arguments (optional days parameter, defaults to 30 days)
            
        Returns:
            Performance summary with key metrics
        """
        try:
            # Check if trading bot and MT5 handler are initialized
            if not hasattr(self, 'trading_bot') or self.trading_bot is None:
                logger.error("Trading bot not initialized")
                return "⚠️ <b>Performance Summary</b>\n\nTrading bot is not initialized."
            
            if not hasattr(self.trading_bot, 'mt5_handler') or self.trading_bot.mt5_handler is None:
                logger.error("MT5 handler not available")
                return "⚠️ <b>Performance Summary</b>\n\nMT5 connection not available. Cannot retrieve performance data."
            
            # Parse days parameter (default to 30 days for performance overview)
            days = 30
            for arg in args:
                if arg.lower().startswith("days="):
                    try:
                        days = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                    
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get trade history using the MT5 handler for accurate data
            deals = self.mt5_handler.get_deals_in_range(start_date, end_date)
            
            if not deals or len(deals) == 0:
                return f"📊 <b>Performance Summary</b>\n\nNo trades found in the last {days} days."
            
            # Group deals by position to analyze complete trades
            position_deals = {}
            for deal in deals:
                if deal.position_id not in position_deals:
                    position_deals[deal.position_id] = []
                position_deals[deal.position_id].append(deal)
            
            # Process each position to calculate metrics
            completed_trades = []
            for position_id, position_deals_list in position_deals.items():
                # Find entry and exit deals
                entry_deals = [d for d in position_deals_list if d.entry == 0]
                exit_deals = [d for d in position_deals_list if d.entry == 1]
                
                # Skip if we don't have both entry and exit deals (open position)
                if not entry_deals or not exit_deals:
                    continue
                    
                entry_deal = sorted(entry_deals, key=lambda d: d.time)[0]
                exit_deal = sorted(exit_deals, key=lambda d: d.time)[-1]
                
                # Calculate profit
                profit = exit_deal.profit
                
                # Create trade entry
                trade = {
                    'id': position_id,
                    'symbol': entry_deal.symbol,
                    'type': 'BUY' if entry_deal.type == 0 else 'SELL',
                    'entry_price': entry_deal.price,
                    'exit_price': exit_deal.price,
                    'profit': profit,
                    'volume': entry_deal.volume,
                    'entry_time': datetime.fromtimestamp(entry_deal.time),
                    'exit_time': datetime.fromtimestamp(exit_deal.time)
                }
                completed_trades.append(trade)
            
            # If no completed trades, return early
            if not completed_trades:
                return f"📊 <b>Performance Summary</b>\n\nNo completed trades found in the last {days} days."
            
            # Calculate metrics
            total_trades = len(completed_trades)
            winning_trades = sum(1 for t in completed_trades if t['profit'] > 0)
            losing_trades = sum(1 for t in completed_trades if t['profit'] < 0)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            total_profit = sum(t['profit'] for t in completed_trades)
            
            # Calculate profit factor
            winning_profits = sum(t['profit'] for t in completed_trades if t['profit'] > 0)
            losing_profits = sum(abs(t['profit']) for t in completed_trades if t['profit'] < 0)
            
            profit_factor = winning_profits / losing_profits if losing_profits > 0 else float('inf')
                
                # Calculate average win and loss
            avg_win = winning_profits / winning_trades if winning_trades > 0 else 0
            avg_loss = losing_profits / losing_trades if losing_trades > 0 else 0
            
            # Calculate drawdown
            # Sort trades by date
            sorted_trades = sorted(completed_trades, key=lambda t: t['entry_time'])
            
            # Calculate running balance and max drawdown
            balance = 0
            peak_balance = 0
            max_drawdown_pct = 0
            
            for trade in sorted_trades:
                balance += trade['profit']
                if balance > peak_balance:
                    peak_balance = balance
                elif peak_balance > 0:
                    drawdown_pct = (peak_balance - balance) / peak_balance * 100
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            
            # Calculate expectancy
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss) if avg_loss > 0 else 0
            
            # Build the performance summary
            summary = [
                f"📊 <b>Performance Summary (Last {days} Days)</b>",
                "",
                "<b>📈 Trade Statistics:</b>",
                f"• Total Trades: <b>{total_trades}</b>",
                f"• Winning Trades: <b>{winning_trades}</b> ({win_rate:.1f}%)",
                f"• Losing Trades: <b>{losing_trades}</b> ({100 - win_rate:.1f}%)",
                "",
                "<b>💰 Profitability:</b>",
                f"• Total Profit: <b>{'+'if total_profit > 0 else ''}{total_profit:.2f}</b>",
                f"• Average Win: <b>+{avg_win:.2f}</b>",
                f"• Average Loss: <b>-{avg_loss:.2f}</b>",
                f"• Profit Factor: <b>{profit_factor:.2f}</b>",
                f"• Expectancy: <b>{'+'if expectancy > 0 else ''}{expectancy:.2f}</b>",
                "",
                "<b>⚠️ Risk Metrics:</b>",
                f"• Max Drawdown: <b>{max_drawdown_pct:.2f}%</b>"
            ]
            
            # Add trade frequency analysis
            trades_per_day = total_trades / days
            summary.append("")
            summary.append("<b>🕒 Trade Frequency:</b>")
            summary.append(f"• Trades per Day: <b>{trades_per_day:.1f}</b>")
            
            # Add top traded symbols if we have enough trades
            if total_trades >= 5:
                symbol_counts = {}
                for trade in completed_trades:
                    symbol = trade['symbol']
                    if symbol not in symbol_counts:
                        symbol_counts[symbol] = {'count': 0, 'profit': 0}
                    symbol_counts[symbol]['count'] += 1
                    symbol_counts[symbol]['profit'] += trade['profit']
                
                # Get top 3 most traded symbols
                top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
                
                if top_symbols:
                    summary.append("")
                    summary.append("<b>🔣 Top Traded Symbols:</b>")
                    for symbol, stats in top_symbols:
                        win_count = sum(1 for t in completed_trades if t['symbol'] == symbol and t['profit'] > 0)
                        symbol_win_rate = (win_count / stats['count']) * 100
                        summary.append(f"• {symbol}: <b>{stats['count']}</b> trades, {symbol_win_rate:.1f}% win rate, P/L: <b>{'+'if stats['profit'] > 0 else ''}{stats['profit']:.2f}</b>")
            
            # Add timestamp
            summary.append("")
            summary.append(f"<i>Analysis from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</i>")
            summary.append(f"<i>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>")
            
            # Add signal quality analysis if we have enough trades
            if total_trades >= 5:
                summary.append("")
                summary.append("<b>📊 Signal Quality Analysis:</b>")
                
                # Calculate average signal quality
                total_quality = 0.0
                trades_with_quality = 0
                high_quality = {"trades": 0, "wins": 0}
                medium_quality = {"trades": 0, "wins": 0}
                low_quality = {"trades": 0, "wins": 0}
                
                for trade in completed_trades:
                    quality = trade.get('signal_quality', 0.0)
                    if quality > 0:
                        total_quality += quality
                        trades_with_quality += 1
                        
                        if quality >= 0.8:  # High quality (80%+)
                            high_quality["trades"] += 1
                            if trade['profit'] > 0:
                                high_quality["wins"] += 1
                        elif quality >= 0.5:  # Medium quality (50-80%)
                            medium_quality["trades"] += 1
                            if trade['profit'] > 0:
                                medium_quality["wins"] += 1
                        else:  # Low quality (<50%)
                            low_quality["trades"] += 1
                            if trade['profit'] > 0:
                                low_quality["wins"] += 1
                
                if trades_with_quality > 0:
                    avg_quality = total_quality / trades_with_quality * 100
                    summary.append(f"• Average Signal Quality: <b>{avg_quality:.1f}%</b>")
                    
                    # High quality stats
                    if high_quality["trades"] > 0:
                        win_rate = high_quality["wins"] / high_quality["trades"] * 100
                        summary.append(f"• High Quality (80%+): <b>{high_quality['trades']}</b> trades, {win_rate:.1f}% win rate")
                    
                    # Medium quality stats
                    if medium_quality["trades"] > 0:
                        win_rate = medium_quality["wins"] / medium_quality["trades"] * 100
                        summary.append(f"• Medium Quality (50-80%): <b>{medium_quality['trades']}</b> trades, {win_rate:.1f}% win rate")
                    
                    # Low quality stats
                    if low_quality["trades"] > 0:
                        win_rate = low_quality["wins"] / low_quality["trades"] * 100
                        summary.append(f"• Low Quality (<50%): <b>{low_quality['trades']}</b> trades, {win_rate:.1f}% win rate")
            
            return "\n".join(summary)
        
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            logger.error(traceback.format_exc())
            return f"⚠️ <b>Error retrieving performance data</b>\n\nPlease try again later. Error: {str(e)[:100]}..."
    
    async def handle_count_command(self, args: List[str]) -> str:
        """
        Handler for the /count command to show trade counts and statistics.
        
        Args:
            args: Command arguments (unused for this command)
            
        Returns:
            Formatted trade count information
        """
        try:
            # Get trade history for analysis
            # Get today, this week, this month, and all-time
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            start_of_month = today.replace(day=1)
            
            # Format dates for history queries
            today_str = today.strftime('%Y-%m-%d')
            week_start_str = start_of_week.strftime('%Y-%m-%d')
            month_start_str = start_of_month.strftime('%Y-%m-%d')
            
            # Get MT5 handler
            if not self.mt5_handler or not self.mt5_handler.connected:
                return "⚠️ <b>MT5 Not Connected</b>\n\nCannot retrieve trade counts without an active MT5 connection."
            
            # Function to count trades in a date range
            def count_trades(from_date, to_date=None):
                """Helper function to count trades from history."""
                if to_date is None:
                    to_date = from_date + timedelta(days=1)
                
                try:
                    # Directly use the provided MT5 handler
                    if hasattr(self.mt5_handler, "get_deals_in_range"):
                        deals = self.mt5_handler.get_deals_in_range(from_date, to_date)
                        if deals is None:
                            logger.warning(f"No deals returned from MT5 handler for the period {from_date} to {to_date}")
                            return 0, 0, 0, 0
                    else:
                        logger.error("MT5 handler does not have 'get_deals_in_range' method.")
                        return 0, 0, 0, 0
                        
                    total_trades = 0
                    winning_trades = 0
                    losing_trades = 0
                    breakeven_trades = 0
                    
                    # Assuming deals is a list of dictionaries with 'profit'
                    if deals:
                        for deal in deals:
                            if deal.entry == 1: # in/out deals
                                continue
                            
                            total_trades += 1
                            if deal.profit > 0:
                                winning_trades += 1
                            elif deal.profit < 0:
                                losing_trades += 1
                            else:
                                breakeven_trades += 1
                                
                    return total_trades, winning_trades, losing_trades, breakeven_trades
                    
                except Exception as e:
                    logger.error(f"Error counting trades: {e}")
                    logger.error(traceback.format_exc())
                    return 0, 0, 0, 0

            # Count trades for today
            today_total, today_win, today_loss, today_even = count_trades(
                datetime.strptime(today_str, '%Y-%m-%d')
            )
            
            # Count trades for the last 7 days
            seven_days_ago = today - timedelta(days=7)
            seven_day_total, seven_day_win, seven_day_loss, seven_day_even = count_trades(
                seven_days_ago,
                today
            )
            
            response = f"""📊 <b>Trade Counts</b>
            
<b>Today ({today_str}):</b>
• Total: {today_total}
• Won: {today_win}
• Lost: {today_loss}
• Breakeven: {today_even}

<b>Last 7 Days:</b>
• Total: {seven_day_total}
• Won: {seven_day_win}
• Lost: {seven_day_loss}
• Breakeven: {seven_day_even}
"""
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving trade counts: {str(e)}")
            logger.error(traceback.format_exc())
            return f"❌ Error retrieving trade counts: {str(e)}"
            
    async def handle_metrics_command(self, args: List[str]) -> str:
        """
        Handler for the /metrics command.
        Shows trading metrics calculated directly from MT5 data.
        
        Args:
            args: Command arguments (optional days parameter, defaults to today)
            
        Returns:
            Trading metrics summary calculated from recent trades
        """
        try:
            # Check if trading bot and MT5 handler are initialized
            if not hasattr(self, 'trading_bot') or self.trading_bot is None:
                logger.error("Trading bot not initialized")
                return "⚠️ <b>Trading Metrics</b>\n\nTrading bot is not initialized."
            
            if not hasattr(self.trading_bot, 'mt5_handler') or self.trading_bot.mt5_handler is None:
                logger.error("MT5 handler not available")
                return "⚠️ <b>Trading Metrics</b>\n\nMT5 connection not available. Cannot retrieve metrics."
            
            # Parse days parameter (default to today)
            days = 1
            for arg in args:
                if arg.lower().startswith("days="):
                    try:
                        days = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                    
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Retrieving metrics for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Get deals for the specified period using the handler
            logger.info(f"Requesting MT5 history deals for {days} day(s)")
            deals = self.mt5_handler.get_deals_in_range(start_date, end_date)
            
            if deals is None:
                logger.error("MT5 handler returned None for get_deals_in_range")
                return f"⚠️ <b>Trading Metrics</b>\n\nFailed to retrieve data from MT5. Please check your connection."
            
            if not deals or len(deals) == 0:
                logger.info(f"No deals found in MT5 for the period ({days} day(s))")
                return f"📏 <b>Trading Metrics</b>\n\nNo trades found in the last {days} day(s). Try increasing the time period with: /metrics days=7"
            
            logger.info(f"Found {len(deals)} deals in MT5 history")
            
            # Group deals by position to analyze complete trades
            position_deals = {}
            for deal in deals:
                if deal.position_id not in position_deals:
                    position_deals[deal.position_id] = []
                position_deals[deal.position_id].append(deal)
            
            logger.info(f"Grouped into {len(position_deals)} unique positions")
            
            # Process each position to calculate metrics
            completed_trades = []
            for position_id, position_deals_list in position_deals.items():
                # Find entry and exit deals
                entry_deals = [d for d in position_deals_list if d.entry == 0]
                exit_deals = [d for d in position_deals_list if d.entry == 1]
                
                # Skip if we don't have both entry and exit deals (open position)
                if not entry_deals or not exit_deals:
                    continue
                    
                entry_deal = sorted(entry_deals, key=lambda d: d.time)[0]
                exit_deal = sorted(exit_deals, key=lambda d: d.time)[-1]
                
                # Calculate profit
                profit = exit_deal.profit
                
                # Create trade entry
                trade = {
                    'id': position_id,
                    'symbol': entry_deal.symbol,
                    'type': 'BUY' if entry_deal.type == 0 else 'SELL',
                    'entry_price': entry_deal.price,
                    'exit_price': exit_deal.price,
                    'profit': profit,
                    'volume': entry_deal.volume,
                    'entry_time': datetime.fromtimestamp(entry_deal.time),
                    'exit_time': datetime.fromtimestamp(exit_deal.time)
                }
                completed_trades.append(trade)
            
            logger.info(f"Found {len(completed_trades)} completed trades out of {len(position_deals)} positions")
            
            # If no completed trades, return early
            if not completed_trades:
                return f"📏 <b>Trading Metrics</b>\n\nNo completed trades found in the last {days} day(s). There may be open positions, but no closed trades. Try increasing the time period with: /metrics days=7"
            
            # Calculate metrics
            total_trades = len(completed_trades)
            winning_trades = sum(1 for t in completed_trades if t['profit'] > 0)
            losing_trades = sum(1 for t in completed_trades if t['profit'] < 0)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            total_profit = sum(t['profit'] for t in completed_trades)
            
            # Calculate profit factor and risk-reward
            winning_profits = sum(t['profit'] for t in completed_trades if t['profit'] > 0)
            losing_profits = sum(abs(t['profit']) for t in completed_trades if t['profit'] < 0)
            
            profit_factor = winning_profits / losing_profits if losing_profits > 0 else float('inf')
            
            # Calculate average win and loss
            avg_win = winning_profits / winning_trades if winning_trades > 0 else 0
            avg_loss = losing_profits / losing_trades if losing_trades > 0 else 0
            
            risk_reward = abs(avg_win / avg_loss) if avg_loss > 0 else float('inf')
            
            # Format metrics into categories
            metrics_sections = {
                "Returns": [
                    f"• Total Profit: <b>{'+'if total_profit > 0 else ''}{total_profit:.2f}</b>",
                    f"• Average Trade: <b>{'+'if total_profit/total_trades > 0 else ''}{total_profit/total_trades:.2f}</b>"
                ],
                "Risk": [
                    f"• Profit Factor: <b>{profit_factor:.2f}</b>",
                    f"• Risk-Reward: <b>{risk_reward:.2f}</b>"
                ],
                "Trade Stats": [
                    f"• Total Trades: <b>{total_trades}</b>",
                    f"• Win Rate: <b>{win_rate:.1f}%</b> ({winning_trades} wins, {losing_trades} losses)"
                ],
                "Averages": [
                    f"• Average Win: <b>+{avg_win:.2f}</b>",
                    f"• Average Loss: <b>-{avg_loss:.2f}</b>"
                ]
            }
            
            # Add symbols breakdown
            symbol_stats = {}
            for trade in completed_trades:
                symbol = trade['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'count': 0, 'profit': 0, 'wins': 0, 'losses': 0}
                
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['profit'] += trade['profit']
                
                if trade['profit'] > 0:
                    symbol_stats[symbol]['wins'] += 1
                elif trade['profit'] < 0:
                    symbol_stats[symbol]['losses'] += 1
            
            if len(symbol_stats) > 0:
                symbols_text = []
                for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                    win_rate = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
                    symbols_text.append(f"• {symbol}: <b>{stats['count']}</b> trades, {win_rate:.1f}% win, P/L: <b>{'+'if stats['profit'] > 0 else ''}{stats['profit']:.2f}</b>")
                
                metrics_sections["Symbols"] = symbols_text[:3]  # Top 3 symbols by count
                
            # Build the final message
            period_str = "today" if days == 1 else f"last {days} days"
            message = [f"📏 <b>Trading Metrics ({period_str})</b>"]
            
            for section, items in metrics_sections.items():
                message.append(f"\n<b>{section}:</b>")
                message.extend(items)
            
            message.append(f"\n<i>Based on {total_trades} completed trades from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</i>")
            message.append(f"<i>For a longer period, try /metrics days=7 or /metrics days=30</i>")
            
            return "\n".join(message)
        except Exception as e:
            logger.error(f"Error generating metrics summary: {str(e)}")
            logger.error(traceback.format_exc())
            return f"⚠️ <b>Error retrieving metrics</b>\n\nPlease try again later. Error: {str(e)[:100]}..."
    
    async def handle_history_command(self, args: List[str]) -> str:
        """
        Handler for the /history command.
        
        Args:
            args: Command arguments (can include date range or "csv" for export)
            
        Returns:
            Formatted trade history information or response with inline keyboard for date selection
        """
        try:
            # Parse command arguments
            export_csv = False
            days = 30  # Default to 30 days
            start_date = None
            end_date = None
            
            # Process arguments
            for arg in args:
                if arg.lower() in ["csv", "export"]:
                    export_csv = True
                elif arg.lower().startswith("days="):
                    try:
                        days = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                elif arg.lower().startswith("from=") or arg.lower().startswith("start="):
                    try:
                        date_str = arg.split("=")[1]
                        start_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except (ValueError, IndexError):
                        pass
                elif arg.lower().startswith("to=") or arg.lower().startswith("end="):
                    try:
                        date_str = arg.split("=")[1]
                        end_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except (ValueError, IndexError):
                        pass
            
            # If no specific date range provided, use days parameter
            if not start_date:
                start_date = datetime.now() - timedelta(days=days)
            if not end_date:
                end_date = datetime.now()
                
            # Check if this is just a request for date selection UI
            if not args:
                # Create an inline keyboard for date selection
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                
                # Predefined periods
                keyboard = [
                    [
                        InlineKeyboardButton("Last 7 days", callback_data="history:days=7"),
                        InlineKeyboardButton("Last 30 days", callback_data="history:days=30")
                    ],
                    [
                        InlineKeyboardButton("Last 60 days", callback_data="history:days=60"),
                        InlineKeyboardButton("Last 90 days", callback_data="history:days=90")
                    ],
                    [
                        InlineKeyboardButton("Export to CSV (30 days)", callback_data="history:days=30:csv")
                    ],
                    [
                        InlineKeyboardButton("Custom range", callback_data="history:custom")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Check if we have a telegram_bot instance to send the reply markup
                if hasattr(self.trading_bot, "telegram_bot") and hasattr(self.trading_bot.telegram_bot, "last_update"):
                    # Get the chat ID from the last update
                    last_update = self.trading_bot.telegram_bot.last_update
                    if last_update is not None and hasattr(last_update, "effective_chat") and last_update.effective_chat is not None:
                        chat_id = last_update.effective_chat.id
                        if self.trading_bot.telegram_bot.bot is not None:
                            await self.trading_bot.telegram_bot.bot.send_message(
                                chat_id=chat_id,
                                text="📅 <b>Select a date range for trade history:</b>",
                                reply_markup=reply_markup,
                                parse_mode="HTML"
                            )
                        return ""  # Return empty string since we sent a separate message
                    else:
                        logger.error("Cannot show history UI: last_update is None or missing effective_chat")
                        # Fall back to the text instructions
                        return ("📊 <b>Trade History Options:</b>\n\n"
                               "Use these commands for trade history:\n"
                               "• <code>/history days=7</code> - Last 7 days\n"
                               "• <code>/history days=30</code> - Last 30 days\n"
                               "• <code>/history days=60</code> - Last 60 days\n"
                               "• <code>/history from=2023-01-01 to=2023-01-31</code> - Custom range\n"
                               "• Add <code>csv</code> to export (e.g., <code>/history days=30 csv</code>)")
                else:
                    # Fallback to text instructions if we can't show the UI
                    logger.warning("Cannot show history UI: telegram_bot or last_update not available")
                    return ("📊 <b>Trade History Options:</b>\n\n"
                           "Use these commands for trade history:\n"
                           "• <code>/history days=7</code> - Last 7 days\n"
                           "• <code>/history days=30</code> - Last 30 days\n"
                           "• <code>/history days=60</code> - Last 60 days\n"
                           "• <code>/history from=2023-01-01 to=2023-01-31</code> - Custom range\n"
                           "• Add <code>csv</code> to export (e.g., <code>/history days=30 csv</code>)")
            
            # Get trade history from MT5 with the selected date range
            trade_history = []
            
            if self.mt5_handler and self.mt5_handler.connected:
                logger.info(f"Fetching MT5 order history from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                # Use the handler to get complete deal information
                try:
                    # Get deals using the handler
                    deals = self.mt5_handler.get_deals_in_range(start_date, end_date)
                    
                    if deals is not None and len(deals) > 0:
                        logger.info(f"Found {len(deals)} deals in MT5 history")
                        
                        # In MT5, deals are more important than orders for analyzing trades
                        # MT5 uses a deal entry system: 
                        # 0 = entry (buy/sell), 1 = exit (closing position)
                        
                        # First, group deals by position_id to get complete trade info
                        position_deals = {}
                        for deal in deals:
                            if deal.position_id not in position_deals:
                                position_deals[deal.position_id] = []
                            position_deals[deal.position_id].append(deal)
                        
                        logger.info(f"Grouped into {len(position_deals)} unique positions")
                        
                        # Process each position to create trade entries
                        for position_id, position_deals_list in position_deals.items():
                            # Skip positions with no deals
                            if len(position_deals_list) == 0:
                                continue
                                
                            # Find entry and exit deals
                            # In MT5, entry=0 means position entry, entry=1 means position exit
                            entry_deals = [d for d in position_deals_list if d.entry == 0]
                            exit_deals = [d for d in position_deals_list if d.entry == 1]
                            
                            # Skip if we don't have both entry and exit deals (open position)
                            if not entry_deals or not exit_deals:
                                continue
                                
                            entry_deal = sorted(entry_deals, key=lambda d: d.time)[0]
                            exit_deal = sorted(exit_deals, key=lambda d: d.time)[-1]
                            
                            # Get the deal type (buy/sell)
                            deal_type = 'BUY' if entry_deal.type == 0 else 'SELL'
                            
                            # Get accurate profit from exit deal
                            profit = exit_deal.profit
                            
                            # Get symbol from deals
                            symbol = entry_deal.symbol
                            
                            # Get entry and exit prices
                            entry_price = entry_deal.price
                            exit_price = exit_deal.price
                            
                            # Extract strategy from comment (if available)
                            strategy = ""
                            if hasattr(entry_deal, 'comment') and entry_deal.comment:
                                # Parse comment field (e.g., SG1:TURTLESOUP:BUY)
                                comment = entry_deal.comment
                                if ":" in comment:
                                    try:
                                        parts = comment.split(":")
                                        if len(parts) >= 2:
                                            strategy = parts[1]
                                    except Exception as e:
                                        logger.warning(f"Error parsing comment: {str(e)}")
                            
                            # Calculate trade duration in minutes
                            duration_minutes = 0
                            try:
                                entry_time = datetime.fromtimestamp(entry_deal.time)
                                exit_time = datetime.fromtimestamp(exit_deal.time)
                                duration = exit_time - entry_time
                                duration_minutes = int(duration.total_seconds() / 60)
                            except Exception as e:
                                logger.warning(f"Error calculating duration: {str(e)}")
                            
                            # Create trade entry with enhanced fields
                            trade_entry = {
                                'id': position_id,
                                'symbol': symbol,
                                'type': deal_type,
                                'entry': entry_price,
                                'exit_price': exit_price,
                                'pnl': profit,
                                'volume': entry_deal.volume,
                                'time': datetime.fromtimestamp(entry_deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                                'exit_time': datetime.fromtimestamp(exit_deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                                'duration_minutes': duration_minutes,
                                'strategy': strategy,
                                'comment': entry_deal.comment if hasattr(entry_deal, 'comment') else ""
                            }
                            trade_history.append(trade_entry)
                        
                        logger.info(f"Created {len(trade_history)} trade history entries from MT5 deals")
                    else:
                        logger.warning("No deals found in MT5 history for the specified date range")
                        
                except Exception as e:
                    logger.error(f"Error fetching deals via handler: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                # Also update the telegram bot's trade history for access elsewhere
                if hasattr(self.trading_bot, "telegram_bot") and hasattr(self.trading_bot.telegram_bot, "trade_history"):
                    self.trading_bot.telegram_bot.trade_history = trade_history
                    
                logger.info(f"Updated trade history with {len(trade_history)} trades from MT5")
            
            # Get history from TelegramBot if available and if we didn't get it from MT5
            if not trade_history and hasattr(self.trading_bot, "telegram_bot") and hasattr(self.trading_bot.telegram_bot, "trade_history"):
                trade_history = self.trading_bot.telegram_bot.trade_history
            
            if not trade_history:
                return "📜 <b>No trade history available</b> for the selected period.\n\nTry a different date range or use /history for interactive options."
            
            # Export to CSV if requested
            if export_csv:
                try:
                    import pandas as pd
                    import os
                    
                    # Create exports directory if it doesn't exist
                    exports_dir = "exports"
                    os.makedirs(exports_dir, exist_ok=True)
                    
                    # Get date range for the filename
                    start_str = start_date.strftime("%Y%m%d")
                    end_str = end_date.strftime("%Y%m%d")
                    csv_filename = f"{exports_dir}/trade_history_{start_str}_to_{end_str}.csv"
                    
                    # Prepare data for CSV export
                    df = pd.DataFrame(trade_history)
                    
                    # Calculate additional statistics per trade
                    if 'entry' in df.columns and 'exit_price' in df.columns:
                        # Calculate pips (account for direction)
                        # First create the pips column for all rows
                        df['pips'] = 0.0
                        
                        # Then calculate pips based on trade direction
                        for idx, row in df.iterrows():
                            if 'type' in df.columns and row['type'] == 'BUY':
                                df.at[idx, 'pips'] = (row['exit_price'] - row['entry']) * 10000
                            else:
                                df.at[idx, 'pips'] = (row['entry'] - row['exit_price']) * 10000
                        
                        # Now that pips are calculated, determine the result for each trade
                        df['result'] = 'BREAK EVEN'  # Default value
                        
                        # Set win/loss based on pips
                        df.loc[df['pips'] > 0, 'result'] = 'WIN'
                        df.loc[df['pips'] < 0, 'result'] = 'LOSS'
                    
                    # Rename comment to strategy if strategy column doesn't exist
                    if 'comment' in df.columns and 'strategy' not in df.columns:
                        df.rename(columns={'comment': 'strategy'}, inplace=True)
                    
                    # Ensure we have all required columns with default values for missing ones
                    required_columns = ['id', 'symbol', 'type', 'entry', 'exit_price', 'pnl', 'pips', 'volume', 
                                      'time', 'exit_time', 'duration_minutes', 'strategy', 'result']
                    
                    for col in required_columns:
                        if col not in df.columns:
                            if col in ['pnl', 'entry', 'exit_price', 'volume', 'duration_minutes', 'pips']:
                                df[col] = 0.0
                            else:
                                df[col] = ""
                    
                    # Reorder columns for the CSV
                    ordered_columns = [col for col in required_columns if col in df.columns] + \
                                    [col for col in df.columns if col not in required_columns]
                    df = df[ordered_columns]
                    
                    # Save to CSV
                    df.to_csv(csv_filename, index=False)
                    
                    # Add summary statistics to a separate sheet in Excel file
                    excel_filename = csv_filename.replace('.csv', '.xlsx')
                    with pd.ExcelWriter(excel_filename) as writer:
                        df.to_excel(writer, sheet_name='Trades', index=False)
                        
                        # Create summary stats
                        summary_data = {
                            'Metric': [
                                'Date Range', 
                                'Total Trades', 
                                'Winning Trades', 
                                'Losing Trades',
                                'Win Rate (%)',
                                'Total Profit/Loss',
                                'Average Profit per Trade',
                                'Average Loss per Trade',
                                'Profit Factor',
                                'Average Trade Duration (min)'
                            ],
                            'Value': [
                                f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                                len(df),
                                len(df[df['result'] == 'WIN']),
                                len(df[df['result'] == 'LOSS']),
                                f"{(len(df[df['result'] == 'WIN']) / len(df) * 100) if len(df) > 0 else 0:.2f}%",
                                f"{df['pnl'].sum():.2f}",
                                f"{df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0:.2f}",
                                f"{df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0:.2f}",
                                f"{abs(df[df['pnl'] > 0]['pnl'].sum()) / abs(df[df['pnl'] < 0]['pnl'].sum()) if abs(df[df['pnl'] < 0]['pnl'].sum()) > 0 else 0:.2f}",
                                f"{df['duration_minutes'].mean():.0f}"
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Create strategy performance sheet if we have strategy data
                        if 'strategy' in df.columns and df['strategy'].any():
                            strategy_stats = df.groupby('strategy').agg({
                                'id': 'count',
                                'pnl': 'sum',
                                'result': lambda x: (x == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
                            }).reset_index()
                            
                            strategy_stats.columns = ['Strategy', 'Trade Count', 'Total P/L', 'Win Rate (%)']
                            strategy_stats = strategy_stats.sort_values(by='Total P/L', ascending=False)
                            strategy_stats.to_excel(writer, sheet_name='Strategy Performance', index=False)
                    
                    logger.info(f"Trade history exported to {csv_filename} and {excel_filename}")
                    
                    # Send the files to the user via Telegram
                    success_msg = f"📊 <b>Trade History Exported</b>\n\nData from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} has been exported.\n\n<i>Files contain {len(df)} trades</i>"
                    
                    # If we have a telegram_bot instance and can access the chat ID, send the files
                    if hasattr(self.trading_bot, "telegram_bot") and self.trading_bot.telegram_bot.last_update:
                        try:
                            # Get the chat ID from the last update
                            last_update = self.trading_bot.telegram_bot.last_update
                            if last_update and hasattr(last_update, "effective_chat") and last_update.effective_chat:
                                chat_id = last_update.effective_chat.id
                                
                                # Send initial message
                                if self.trading_bot.telegram_bot.bot is not None:
                                    await self.trading_bot.telegram_bot.bot.send_message(
                                        chat_id=chat_id,
                                        text="⏳ <b>Sending files, please wait...</b>",
                                        parse_mode='HTML'
                                    )
                                else:
                                    logger.error("Telegram bot instance is None, cannot send message.")
                                
                                # Send CSV file
                                if self.trading_bot.telegram_bot.bot is not None:
                                    with open(csv_filename, 'rb') as csv_file:
                                        await self.trading_bot.telegram_bot.bot.send_document(
                                            chat_id=chat_id,
                                            document=csv_file,
                                            filename=os.path.basename(csv_filename),
                                            caption="📊 Trade History (CSV format)"
                                        )
                                else:
                                    logger.error("Telegram bot instance is None, cannot send CSV document.")
                                
                                # Send Excel file
                                if self.trading_bot.telegram_bot.bot is not None:
                                    with open(excel_filename, 'rb') as excel_file:
                                        await self.trading_bot.telegram_bot.bot.send_document(
                                            chat_id=chat_id,
                                            document=excel_file,
                                            filename=os.path.basename(excel_filename),
                                            caption="📈 Trade History with Summary (Excel format)"
                                        )
                                else:
                                    logger.error("Telegram bot instance is None, cannot send Excel document.")
                                
                                logger.info(f"Successfully sent history files to chat ID: {chat_id}")
                                return success_msg + "\n\n✅ <b>Files have been sent to this chat.</b>"
                            else:
                                logger.error("Cannot send files: invalid update or missing chat ID")
                        except Exception as file_e:
                            logger.error(f"Error sending files via Telegram: {str(file_e)}")
                            logger.error(traceback.format_exc())
                            return success_msg + f"\n\n⚠️ <b>Could not send files directly:</b> {str(file_e)}\n\nFiles are saved at:\n• <code>{csv_filename}</code>\n• <code>{excel_filename}</code>"
                    
                    # Default message if we can't send the files
                    return f"📊 <b>Trade History Exported</b>\n\nData from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} has been exported:\n\n• CSV file: <code>{csv_filename}</code>\n• Excel file with summary: <code>{excel_filename}</code>\n\n<i>Files contain {len(df)} trades</i>"
                    
                except Exception as e:
                    logger.error(f"Error exporting trade history to CSV: {str(e)}")
                    logger.error(traceback.format_exc())
                    return f"⚠️ <b>Error exporting trade history</b>\n\nFailed to create export files: {str(e)}"
            
            # Sort trades by time (newest first)
            trade_history.sort(key=lambda x: x.get('time', ''), reverse=True)
            
            # Validate trade data
            valid_trades = []
            for trade in trade_history:
                if ('pnl' in trade or 'profit' in trade) and 'symbol' in trade:
                    # Normalize trade data
                    if 'profit' in trade and 'pnl' not in trade:
                        trade['pnl'] = trade['profit']
                    
                    # Make sure we have numeric values
                    for field in ['entry', 'exit_price', 'pnl']:
                        if field in trade and not isinstance(trade[field], (int, float)):
                            try:
                                trade[field] = float(trade[field])
                            except (ValueError, TypeError):
                                trade[field] = 0.0
                    
                    valid_trades.append(trade)
                else:
                    logger.warning(f"Skipping invalid trade record: {trade}")
            
            if not valid_trades:
                return "⚠️ <b>No valid trade data found</b>\n\nTrades exist but lack complete information."
            
            # Limit to display 10 trades in the message to avoid it being too long
            display_trades = valid_trades[:10]
            total_trades = len(valid_trades)
            
            # Calculate summary stats
            winning_count = sum(1 for trade in valid_trades if trade.get('pnl', 0) > 0)
            losing_count = sum(1 for trade in valid_trades if trade.get('pnl', 0) < 0)
            break_even_count = sum(1 for trade in valid_trades if trade.get('pnl', 0) == 0)
            total_pnl = sum(trade.get('pnl', 0) for trade in valid_trades)
            
            # Calculate win rate and average trade
            win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor if there are losing trades
            total_wins = sum(trade.get('pnl', 0) for trade in valid_trades if trade.get('pnl', 0) > 0)
            total_losses = sum(abs(trade.get('pnl', 0)) for trade in valid_trades if trade.get('pnl', 0) < 0)
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Create summary header with improved stats
            history_text = f"""📈 <b>TRADE HISTORY SUMMARY</b> 📉

<b>📆 Period:</b> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

<b>📊 Performance Metrics:</b>
• Total Trades: <b>{total_trades}</b> ({winning_count} wins, {losing_count} losses, {break_even_count} even)
• Win Rate: <b>{win_rate:.1f}%</b>
• Net P/L: <b>{"+" if total_pnl > 0 else ""}{total_pnl:.2f}</b>
• Avg Trade: <b>{"+" if avg_trade > 0 else ""}{avg_trade:.2f}</b>
• Profit Factor: <b>{profit_factor:.2f}</b>

<i>Showing recent {len(display_trades)} of {total_trades} trades. Use /history csv for export.</i>
"""

            # Add strategy breakdown if we have strategy data
            strategies = {}
            for trade in valid_trades:
                if 'strategy' in trade and trade['strategy']:
                    if trade['strategy'] not in strategies:
                        strategies[trade['strategy']] = {
                            'count': 0,
                            'wins': 0,
                            'losses': 0,
                            'pnl': 0
                        }
                    
                    strategies[trade['strategy']]['count'] += 1
                    if trade.get('pnl', 0) > 0:
                        strategies[trade['strategy']]['wins'] += 1
                    elif trade.get('pnl', 0) < 0:
                        strategies[trade['strategy']]['losses'] += 1
                    strategies[trade['strategy']]['pnl'] += trade.get('pnl', 0)
            
            if strategies:
                history_text += "\n<b>📋 Strategy Performance:</b>\n"
                for strategy, stats in sorted(strategies.items(), key=lambda x: x[1]['pnl'], reverse=True):
                    win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                    history_text += f"• {strategy}: <b>{"+" if stats['pnl'] > 0 else ""}{stats['pnl']:.2f}</b> ({stats['count']} trades, {win_rate:.0f}% win)\n"
            
            history_text += "\n<b>🔄 Recent Trades:</b>\n"
            
            # Add each trade with improved formatting
            for i, trade in enumerate(display_trades, 1):
                # Get trade values with defaults for missing data
                pnl = trade.get('pnl', 0)
                symbol = trade.get('symbol', 'Unknown')
                trade_type = trade.get('type', 'Unknown')
                entry = trade.get('entry', 0.0)
                exit_price = trade.get('exit_price', 0.0) or trade.get('exit', 0.0) or trade.get('price_current', 0.0)
                volume = trade.get('volume', 0.0)
                strategy = trade.get('strategy', '')
                
                # Better emojis based on profit and direction
                if pnl > 0:
                    result_icon = "💰"  # Money bag for profitable trades
                elif pnl < 0:
                    result_icon = "📉"  # Downward trend for losing trades 
                else:
                    result_icon = "⚖️"  # Balance scale for break-even
                
                # Direction icons
                if isinstance(trade_type, str) and trade_type.lower() in ['buy', 'long']:
                    direction_icon = "🔼"  # Up triangle for buy/long
                elif isinstance(trade_type, str) and trade_type.lower() in ['sell', 'short']:
                    direction_icon = "🔽"  # Down triangle for sell/short
                else:
                    direction_icon = "◾"  # Square for unknown
                
                # Calculate profit percentage
                profit_pct = ""
                if entry != 0 and exit_price != 0 and abs(entry - exit_price) > 0.00001:
                    if isinstance(trade_type, str) and trade_type.lower() == 'buy':
                        pct = (exit_price - entry) / entry * 100
                    else:  # For sell orders, the calculation is reversed
                        pct = (entry - exit_price) / entry * 100
                    pct_sign = "+" if pct > 0 else ""
                    profit_pct = f"{pct_sign}{pct:.2f}%"
                
                # Calculate pips
                pips = 0
                if isinstance(trade_type, str) and trade_type.lower() == 'buy':
                    pips = (exit_price - entry) * 10000
                else:
                    pips = (entry - exit_price) * 10000
                pips_text = f"{pips:.1f} pips"
                
                # Format trade duration
                duration = ""
                if 'duration_minutes' in trade:
                    mins = trade['duration_minutes']
                    if mins >= 1440:  # More than a day
                        days = mins // 1440
                        hours = (mins % 1440) // 60
                        duration = f"{days}d {hours}h"
                    elif mins >= 60:  # More than an hour
                        hours = mins // 60
                        minutes = mins % 60
                        duration = f"{hours}h {minutes}m"
                    else:
                        duration = f"{mins}m"
                elif 'exit_time' in trade and trade.get('exit_time') and trade.get('time'):
                    try:
                        entry_dt = datetime.strptime(trade.get('time'), '%Y-%m-%d %H:%M:%S')
                        exit_dt = datetime.strptime(trade.get('exit_time'), '%Y-%m-%d %H:%M:%S')
                        duration_td = exit_dt - entry_dt
                        
                        if duration_td.days > 0:
                            duration = f"{duration_td.days}d {duration_td.seconds//3600}h"
                        elif duration_td.seconds > 3600:
                            duration = f"{duration_td.seconds//3600}h {(duration_td.seconds%3600)//60}m"
                        else:
                            duration = f"{duration_td.seconds//60}m"
                    except:
                        pass
                
                # Trade time in readable format
                trade_time = trade.get('time', '')
                if trade_time:
                    try:
                        dt = datetime.strptime(trade_time, '%Y-%m-%d %H:%M:%S')
                        trade_time = dt.strftime('%d %b %H:%M')  # e.g., "15 Jan 14:30"
                    except:
                        pass
                
                # Clean layout with clear sections
                history_text += f"""<b>{i}. {result_icon} {symbol}</b> {direction_icon} {volume:.2f}
   P/L: <b>{"+" if pnl > 0 else ""}{pnl:.2f}</b> ({profit_pct}) [{pips_text}]
   {trade_type} {entry:.5f} → {exit_price:.5f} | {duration}
   {f"📊 Quality: {trade.get('signal_quality', 0.0)*100:.1f}%" if 'signal_quality' in trade else ""}
   {f"📝 {strategy}" if strategy else ""} | {trade_time}
"""
            
            # Add footer with navigation options
            history_text += f"""
<i>For different date ranges or CSV export:</i>
• /history days=7
• /history days=60 csv
• /history from=2023-01-01 to=2023-01-31
"""
            
            return history_text
            
        except Exception as e:
            logger.error(f"Error retrieving trade history: {str(e)}")
            logger.error(traceback.format_exc())
            return f"⚠️ <b>Error retrieving trade history</b>\n\nPlease try again later. Error: {str(e)[:100]}..." 

    async def register_handlers(self):
        """
        Register command handlers with the trading bot's Telegram instance.
        This method is called during initialization to set up command handlers.
        """
        logger.info("Registering command handlers via register_handlers method")
        
        if not self.trading_bot or not hasattr(self.trading_bot, 'telegram_bot') or not self.trading_bot.telegram_bot:
            logger.warning("Cannot register handlers: Telegram bot not available in trading_bot")
            return False
            
        # Register all commands with the Telegram bot
        await self.register_all_commands(self.trading_bot.telegram_bot)
        
        return True