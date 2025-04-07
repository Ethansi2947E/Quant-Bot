import traceback
from typing import Dict, Any, Tuple
from datetime import datetime
from src.mt5_handler import MT5Handler
from src.risk_manager import RiskManager
from src.telegram.telegram_bot import TelegramBot
from loguru import logger

class PositionManager:
    """
    Handles position management functionality for the trading bot.
    
    This class is responsible for:
    - Managing open trades (stop loss, take profit, trailing stop)
    - Updating trade records in the database
    - Closing pending trades
    - Reconciling trade records with MT5 platform data
    """
    
    def __init__(self, mt5_handler=None, risk_manager=None, telegram_bot=None, config=None):
        """
        Initialize the PositionManager.
        
        Args:
            mt5_handler: MetaTrader 5 handler instance
            risk_manager: Risk manager instance
            telegram_bot: Telegram bot instance
            config: Configuration dictionary
        """
        self.mt5_handler = mt5_handler if mt5_handler else MT5Handler()
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        self.telegram_bot = telegram_bot if telegram_bot else TelegramBot.get_instance()
        self.config = config or {}
        
        # State tracking
        self.active_trades = {}
        self.trailing_stop_data = {}
        self.trailing_stop_enabled = self.config.get('use_trailing_stop', True)
        
    def set_mt5_handler(self, mt5_handler):
        """Set the MT5Handler instance after initialization."""
        self.mt5_handler = mt5_handler
        
    def set_risk_manager(self, risk_manager):
        """Set the RiskManager instance after initialization."""
        self.risk_manager = risk_manager
        
    def set_telegram_bot(self, telegram_bot):
        """Set the TelegramBot instance after initialization."""
        self.telegram_bot = telegram_bot
        
    def set_config(self, config):
        """Set the configuration dictionary after initialization."""
        self.config = config
        
    async def manage_open_trades(self) -> None:
        """
        Manage all currently open trading positions.
        
        This method:
        - Updates stop loss and take profit levels
        - Applies trailing stops
        - Checks for take profit/stop loss hits
        - Executes partial closures if conditions are met
        """
        if not self.mt5_handler:
            logger.warning("MT5Handler not set, cannot manage open trades")
            return
            
        try:
            # Get all open positions
            positions = self.mt5_handler.get_open_positions()
            
            if not positions:
                return
                
            logger.debug(f"Managing {len(positions)} open positions")
            
            for position in positions:
                try:
                    # Extract position details
                    symbol = position.get("symbol", "")
                    ticket = position.get("ticket", 0)
                    position_type = self._get_position_type(position)
                    current_sl = position.get("sl", 0.0)
                    current_tp = position.get("tp", 0.0)
                    entry_price = position.get("price_open", 0.0)
                    current_price = position.get("price_current", 0.0)
                    profit = position.get("profit", 0.0)
                    
                    # Apply trailing stop if enabled
                    if self.trailing_stop_enabled:
                        self._apply_trailing_stop(position)
                    
                    # Update position in database or tracking system
                    self._update_trade_in_database(position)
                    
                    # TODO: Implement partial closure logic
                    
                except Exception as e:
                    logger.error(f"Error managing position {position.get('ticket', 'unknown')}: {str(e)}")
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Error in manage_open_trades: {str(e)}")
            logger.error(traceback.format_exc())

    def _update_trade_in_database(self, position: Dict[str, Any]) -> None:
        """
        Update a trade record in the database.
        
        Args:
            position: Position data dictionary
        """
        # Implementation depends on the database system being used
        # For now, we'll just update the in-memory tracking
        ticket = position.get("ticket", 0)
        if ticket:
            self.active_trades[ticket] = position
            
    def _get_position_type(self, position: Dict[str, Any]) -> str:
        """
        Get the position type (buy/sell) from a position dictionary.
        
        Args:
            position: Position data dictionary
            
        Returns:
            String indicating position type ("buy" or "sell")
        """
        position_type = position.get("type", 0)
        return "buy" if position_type == 0 else "sell"
        
    def _apply_trailing_stop(self, position: Dict[str, Any]) -> bool:
        """
        Apply trailing stop to a position if conditions are met.
        
        Args:
            position: Position data dictionary
            
        Returns:
            Boolean indicating if trailing stop was applied
        """
        if not self.trailing_stop_enabled:
            return False
            
        try:
            ticket = position.get("ticket", 0)
            symbol = position.get("symbol", "")
            
            # Early exit if no symbol or ticket
            if not symbol or not ticket:
                logger.warning(f"Missing symbol or ticket in position data")
                return False
                
            position_type = self._get_position_type(position)
            current_sl = position.get("sl", 0.0)
            entry_price = position.get("price_open", 0.0)
            current_price = position.get("price_current", 0.0)
            
            # Get trailing stop settings
            trailing_settings = self.config.get("trailing_stop_settings", {})
            trail_step_pips = trailing_settings.get("trail_step_pips", 5)
            
            # Calculate pip value based on symbol properties
            symbol_info = self.mt5_handler.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"Cannot get symbol info for {symbol}")
                return False
                
            # Calculate pip value manually from symbol properties
            digits = getattr(symbol_info, "digits", 5)
            pip_value = 0.0001 if digits == 4 else 0.00001 if digits == 5 else 0.000001
            
            # Calculate trailing step size
            trail_step = trail_step_pips * pip_value
            
            # Initialize tracking if needed
            if ticket not in self.trailing_stop_data:
                # Calculate activation price
                activation_price = self._calculate_activation_price(
                    position_type, entry_price, current_sl
                )
                
                self.trailing_stop_data[ticket] = {
                    "activation_price": activation_price,
                    "highest_price": current_price if position_type == "buy" else float('inf'),
                    "lowest_price": current_price if position_type == "sell" else float('-inf'),
                    "last_update": datetime.now(),
                }
                
                logger.debug(
                    f"Initialized trailing stop for {symbol} #{ticket}: "
                    f"Activation @ {activation_price}, Current @ {current_price}"
                )
                return False
                
            # Retrieve tracking data
            tracking_data = self.trailing_stop_data[ticket]
            activation_price = tracking_data["activation_price"]
            
            # Check if activation price has been reached
            if position_type == "buy":
                if current_price >= activation_price:
                    # Update highest observed price
                    if current_price > tracking_data["highest_price"]:
                        tracking_data["highest_price"] = current_price
                        
                    # Calculate new stop loss based on trailing distance
                    new_sl = tracking_data["highest_price"] - trail_step
                    
                    # Only move stop loss up, never down
                    if new_sl > current_sl + (0.1 * pip_value):  # Small buffer to avoid unnecessary updates
                        # Request stop loss modification using documented modify_position method
                        success = self.mt5_handler.modify_position(
                            ticket=ticket,
                            new_sl=new_sl,
                            new_tp=position.get("tp", 0.0)
                        )
                        
                        if success:
                            logger.info(
                                f"Updated trailing stop for {symbol} #{ticket} "
                                f"from {current_sl} to {new_sl}"
                            )
                            tracking_data["last_update"] = datetime.now()
                            return True
            
            elif position_type == "sell":
                if current_price <= activation_price:
                    # Update lowest observed price
                    if current_price < tracking_data["lowest_price"]:
                        tracking_data["lowest_price"] = current_price
                        
                    # Calculate new stop loss based on trailing distance
                    new_sl = tracking_data["lowest_price"] + trail_step
                    
                    # Only move stop loss down, never up
                    if new_sl < current_sl - (0.1 * pip_value):  # Small buffer to avoid unnecessary updates
                        # Request stop loss modification using documented modify_position method
                        success = self.mt5_handler.modify_position(
                            ticket=ticket,
                            new_sl=new_sl,
                            new_tp=position.get("tp", 0.0)
                        )
                        
                        if success:
                            logger.info(
                                f"Updated trailing stop for {symbol} #{ticket} "
                                f"from {current_sl} to {new_sl}"
                            )
                            tracking_data["last_update"] = datetime.now()
                            return True
            
        except Exception as e:
            logger.error(f"Error applying trailing stop to position {position.get('ticket', 'unknown')}: {str(e)}")
            
        return False
        
    def _calculate_activation_price(self, direction: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate the activation price for a trailing stop.
        
        Args:
            direction: Trade direction ("buy" or "sell")
            entry_price: Entry price
            stop_loss: Initial stop loss price
            
        Returns:
            Activation price for the trailing stop
        """
        risk = abs(entry_price - stop_loss)
        activation_factor = self.config.get("trailing_activation_factor", 1.0)
        
        if direction == "buy":
            return entry_price + (risk * activation_factor)
        else:
            return entry_price - (risk * activation_factor)
            
    async def close_pending_trades(self) -> Tuple[int, int]:
        """
        Close all pending trades.
        
        Returns:
            Tuple of (success_count, failed_count)
        """
        if not self.mt5_handler:
            logger.warning("MT5Handler not set, cannot close pending trades")
            return (0, 0)
            
        success_count = 0
        failed_count = 0
        
        try:
            positions = self.mt5_handler.get_open_positions()
            
            if not positions:
                logger.info("No open positions to close")
                return (0, 0)
                
            logger.info(f"Closing {len(positions)} open positions")
            
            for position in positions:
                ticket = position.get("ticket", 0)
                symbol = position.get("symbol", "")
                
                if not ticket or not symbol:
                    logger.warning(f"Invalid position data: {position}")
                    failed_count += 1
                    continue
                    
                # Close the position
                result = self.mt5_handler.close_position(ticket)
                
                if result == True:
                    success_count += 1
                    logger.info(f"Successfully closed position {ticket} on {symbol}")
                    
                    # Send notification if telegram is available
                    if self.telegram_bot:
                        try:
                            await self.telegram_bot.send_trade_update(
                                ticket=ticket,
                                symbol=symbol,
                                action="CLOSED",
                                price=position.get("price_current", 0.0),
                                profit=position.get("profit", 0.0),
                                reason="Bot shutdown"
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
            
        return (success_count, failed_count)
        
    async def reconcile_trades(self) -> None:
        """
        Reconcile local trade records with MT5 platform data.
        
        This method:
        - Identifies discrepancies between local records and MT5
        - Updates local records to match MT5 state
        - Handles orphaned trades that exist in MT5 but not in local records
        """
        if not self.mt5_handler:
            logger.warning("MT5Handler not set, cannot reconcile trades")
            return
            
        try:
            # Get all open positions from MT5
            mt5_positions = self.mt5_handler.get_open_positions()
            
            if not mt5_positions:
                # If no positions in MT5, clear local tracking
                self.active_trades = {}
                return
                
            # Create dictionary of MT5 positions by ticket
            mt5_positions_dict = {pos.get("ticket"): pos for pos in mt5_positions if pos.get("ticket")}
            
            # Remove closed positions from local tracking
            closed_tickets = []
            for ticket in self.active_trades.keys():
                if ticket not in mt5_positions_dict:
                    closed_tickets.append(ticket)
                    
            for ticket in closed_tickets:
                logger.info(f"Removing closed position {ticket} from local tracking")
                if ticket in self.active_trades:
                    del self.active_trades[ticket]
                if ticket in self.trailing_stop_data:
                    del self.trailing_stop_data[ticket]
                    
            # Add new positions to local tracking
            for ticket, position in mt5_positions_dict.items():
                if ticket not in self.active_trades:
                    logger.info(f"Adding new position {ticket} to local tracking")
                    self.active_trades[ticket] = position
                else:
                    # Update existing position
                    self.active_trades[ticket] = position
                    
        except Exception as e:
            logger.error(f"Error reconciling trades: {str(e)}")
            logger.error(traceback.format_exc())
                
    async def _notify_trade_action(self, message: str) -> None:
        """
        Send notification about a trade action.
        
        Args:
            message: Notification message
        """
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_notification(message)
            except Exception as e:
                logger.warning(f"Failed to send notification: {str(e)}")
                
    async def update_positions(self, symbol: str, current_price: float) -> None:
        """
        Update positions with the latest price data for a specific symbol.
        
        This method is called by the trading bot when new tick/price data is received.
        It updates position tracking and applies position management rules.
        
        Args:
            symbol: The trading symbol (e.g., 'EURUSD')
            current_price: The current bid price
        """
        if not self.mt5_handler:
            logger.warning("MT5Handler not set, cannot update positions")
            return
            
        try:
            # Get open positions for this symbol
            all_positions = self.mt5_handler.get_open_positions()
            if not all_positions:
                logger.debug(f"No open positions found for any symbol")
                return
                
            # Filter for the specific symbol
            positions = [p for p in all_positions if p.get("symbol") == symbol]
            
            if not positions:
                logger.debug(f"No open positions found for {symbol}")
                return
                
            logger.info(f"Updating {len(positions)} positions for {symbol} at price {current_price}")
            
            for position in positions:
                ticket = position.get("ticket", 0)
                position_type = self._get_position_type(position)
                entry_price = position.get("price_open", 0.0)
                
                # Skip invalid positions
                if not ticket:
                    logger.warning(f"Found position with no ticket for {symbol}, skipping")
                    continue
                    
                # Add detailed logging
                logger.info(f"Position #{ticket}: Type={position_type}, Entry={entry_price}, Current={current_price}")
                
                # Update the position in our tracking
                if ticket in self.active_trades:
                    # Update current price in our tracked position
                    self.active_trades[ticket]["price_current"] = current_price
                    logger.debug(f"Updated tracked position #{ticket} with current price {current_price}")
                    
                    # Calculate current profit
                    volume = position.get("volume", 0.0)
                    profit_pips = (current_price - entry_price) * 10000 if position_type == "buy" else (entry_price - current_price) * 10000
                    logger.debug(f"Position #{ticket} current P/L: {profit_pips:.1f} pips")
                    
                    # Apply trailing stop logic if enabled
                    if self.trailing_stop_enabled and ticket in self.trailing_stop_data:
                        self._apply_trailing_stop(position)
                        logger.debug(f"Applied trailing stop to position #{ticket}")
                else:
                    # Position not in our tracking, add it
                    logger.info(f"Adding position #{ticket} to tracking")
                    self.active_trades[ticket] = position
                
        except Exception as e:
            logger.error(f"Error updating positions for {symbol}: {str(e)}")
            logger.error(traceback.format_exc()) 