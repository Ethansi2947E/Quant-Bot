import traceback
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import time
from datetime import datetime, timedelta
import asyncio

from src.telegram.telegram_bot import TelegramBot
from src.mt5_handler import MT5Handler
from src.utils.market_utils import calculate_pip_value, convert_pips_to_price
from src.risk_manager import RiskManager

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
                logger.debug("No open positions to manage")
                return
                
            # Check trailing stop enabled flag from config
            from config.config import TRADE_EXIT_CONFIG
            trailing_enabled = TRADE_EXIT_CONFIG.get('trailing_stop', {}).get('enabled', True)
            
            # Use the most up-to-date setting
            self.trailing_stop_enabled = trailing_enabled
            
            logger.info(f"Managing {len(positions)} open positions with trailing stop {'enabled' if self.trailing_stop_enabled else 'disabled'}")
            
            # Cache of latest tick data to avoid repeated fetches for the same symbol
            tick_cache = {}
            
            for position in positions:
                try:
                    # Extract position details
                    symbol = position.get("symbol", "")
                    ticket = position.get("ticket", 0)
                    
                    if not symbol or not ticket:
                        logger.warning(f"Skipping position with missing symbol or ticket: {position}")
                        continue
                    
                    position_type = self._get_position_type(position)
                    current_sl = position.get("sl", 0.0)
                    current_tp = position.get("tp", 0.0)
                    entry_price = position.get("open_price", 0.0)
                    current_price = position.get("current_price", 0.0)
                    profit = position.get("profit", 0.0)
                    
                    # Defensive: Warn if entry_price is 0.0
                    if entry_price == 0.0:
                        logger.warning(f"Position {symbol} #{ticket} has open_price=0.0! This will break trailing stop logic.")
                    
                    # Ensure we have accurate price data
                    if current_price == 0.0:
                        # Get latest tick data for this symbol (using cache if available)
                        if symbol not in tick_cache:
                            tick_cache[symbol] = self.mt5_handler.get_last_tick(symbol)
                            
                        tick = tick_cache[symbol]
                        if tick:
                            # Use appropriate price based on position type
                            current_price = tick.get('ask', 0.0) if position_type == "buy" else tick.get('bid', 0.0)
                            
                            # If still zero, try accessing as dictionary with bracket notation
                            if current_price == 0.0 and isinstance(tick, dict):
                                current_price = tick['ask'] if position_type == "buy" and 'ask' in tick else tick['bid'] if 'bid' in tick else 0.0
                                
                            position["current_price"] = current_price
                            logger.debug(f"Updated {symbol} #{ticket} with current price {current_price}")
                    
                    logger.debug(f"Managing position #{ticket}: {symbol} {position_type} at {entry_price}, Current: {current_price}, SL: {current_sl}, TP: {current_tp}, P/L: {profit}")
                    
                    # Apply trailing stop if enabled
                    if self.trailing_stop_enabled:
                        try:
                            trailing_applied = self._apply_trailing_stop(position)
                            if trailing_applied:
                                logger.info(f"Trailing stop applied for position #{ticket}: {symbol}")
                        except Exception as e:
                            logger.error(f"Error applying trailing stop for position #{ticket}: {symbol} - {str(e)}")
                            logger.error(traceback.format_exc())
                    
                    # Update position in database or tracking system
                    self._update_trade_in_database(position)
                    
                    # Implement additional position management logic here if needed
                    
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
            # Extract position details
            ticket = position.get("ticket", 0)
            symbol = position.get("symbol", "")
            position_type = self._get_position_type(position)
            entry_price = position.get("open_price", 0.0)
            current_price = position.get("current_price", 0.0)
            current_sl = position.get("sl", 0.0)
            
            # Get the trailing step size
            from config.config import TRADE_EXIT_CONFIG
            # Use trail_points from config (new name) but fall back to trail_step_pips (old name) for backward compatibility
            trail_step_pips = TRADE_EXIT_CONFIG.get('trailing_stop', {}).get('trail_points', 
                             TRADE_EXIT_CONFIG.get('trailing_stop', {}).get('trail_step_pips', 20))
            
            # Get pip value for this symbol using the utility function
            pip_value = calculate_pip_value(symbol, mt5_handler=self.mt5_handler)
            
            # Calculate trailing step in price
            trail_step = trail_step_pips * pip_value
            
            # Get minimum stop level from MT5
            min_stop_distance = self.mt5_handler.get_min_stop_distance(symbol)
            
            logger.debug(f"Minimum stop distance for {symbol}: {min_stop_distance}")
            
            # Initialize tracking if needed
            if ticket not in self.trailing_stop_data:
                # Check if stop loss is set
                if current_sl == 0.0:
                    logger.warning(f"Position {symbol} #{ticket} has no stop loss (SL=0.0), setting default stop loss")
                    # Calculate a default stop loss based on current price
                    if position_type == "buy":
                        # For buys, set stop loss 2% below entry
                        default_sl = entry_price * 0.98
                    else:
                        # For sells, set stop loss 2% above entry
                        default_sl = entry_price * 1.02
                    
                    # Try to modify the position with a default stop loss
                    success = self.mt5_handler.modify_position(
                        ticket=ticket,
                        new_sl=default_sl,
                        new_tp=position.get("tp", 0.0)
                    )
                    if success:
                        logger.info(f"Set default stop loss for {symbol} #{ticket} at {default_sl}")
                        current_sl = default_sl
                    else:
                        logger.warning(f"Failed to set default stop loss for {symbol} #{ticket}")
                    
                # Calculate activation price
                activation_price = self._calculate_activation_price(
                    position_type, entry_price, current_sl
                )
                
                # If activation price is based on a zero stop loss, it may be invalid
                if current_sl == 0.0 and activation_price == 0.0:
                    logger.warning(f"Invalid activation price (0.0) for {symbol} #{ticket}, trailing stop will not work")
                
                self.trailing_stop_data[ticket] = {
                    "activation_price": activation_price,
                    "highest_price": current_price if position_type == "buy" else float('inf'),
                    "lowest_price": current_price if position_type == "sell" else float('-inf'),
                    "last_update": datetime.now(),
                    "sl_set": current_sl > 0,  # Track if SL is set
                    "initial_sl": current_sl,  # Store initial SL value
                    "position_type": position_type,  # Store position type for reference
                }
                
                logger.info(
                    f"Initialized trailing stop for {symbol} #{ticket}: "
                    f"Activation @ {activation_price}, Current @ {current_price}, "
                    f"Entry @ {entry_price}, SL @ {current_sl}, "
                    f"Trail Step: {trail_step_pips} pips, "
                    f"Direction: {position_type}"
                )
                return False
                
            # Retrieve tracking data
            tracking_data = self.trailing_stop_data[ticket]
            activation_price = tracking_data["activation_price"]
            
            # Log current state for debugging
            logger.debug(
                f"Trailing stop check for {symbol} #{ticket}: "
                f"Type: {position_type}, "
                f"Current: {current_price}, "
                f"Activation: {activation_price}, "
                f"Current SL: {current_sl}, "
                f"{'Highest' if position_type == 'buy' else 'Lowest'} Price: {tracking_data['highest_price'] if position_type == 'buy' else tracking_data['lowest_price']}"
            )
            
            # Check if activation price has been reached - fix condition comparison for buy orders
            if position_type == "buy":
                has_activated = current_price >= activation_price
                
                if has_activated:
                    logger.info(f"Buy trailing stop activated for {symbol} #{ticket}: price {current_price} >= activation {activation_price}")
                    
                    # Update highest observed price
                    if current_price > tracking_data["highest_price"]:
                        tracking_data["highest_price"] = current_price
                        logger.debug(f"Updated highest price for {symbol} #{ticket} to {current_price}")
                        
                    # Calculate new stop loss based on trailing distance
                    new_sl = tracking_data["highest_price"] - trail_step
                    
                    # Ensure the stop loss respects minimum distance
                    min_valid_sl = current_price - min_stop_distance
                    
                    # For buy positions, stop loss must be below current price but not too close
                    # If new SL is too close to current price (above min_valid_sl), adjust it down
                    if new_sl > min_valid_sl:
                        logger.debug(f"Adjusted buy SL from {new_sl} to {min_valid_sl} to respect minimum distance ({min_stop_distance})")
                        new_sl = min_valid_sl
                    
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
                                f"from {current_sl} to {new_sl} (moved UP by {new_sl - current_sl:.5f})"
                            )
                            tracking_data["last_update"] = datetime.now()
                            return True
                        else:
                            logger.warning(
                                f"Failed to update trailing stop for {symbol} #{ticket} "
                                f"from {current_sl} to {new_sl}"
                            )
                    else:
                        logger.debug(
                            f"No trailing stop update needed for {symbol} #{ticket}: "
                            f"new SL {new_sl} not significantly higher than current SL {current_sl}"
                        )
                else:
                    # Periodically log that activation price hasn't been reached yet
                    last_check = tracking_data.get("last_check_time")
                    now = datetime.now()
                    if not last_check or (now - last_check).total_seconds() > 1800:  # Log once per 30 minutes
                        logger.debug(f"Waiting for activation: {symbol} #{ticket} price {current_price} needs to reach {activation_price}")
                        tracking_data["last_check_time"] = now
            
            elif position_type == "sell":
                has_activated = current_price <= activation_price
                
                if has_activated:
                    logger.info(f"Sell trailing stop activated for {symbol} #{ticket}: price {current_price} <= activation {activation_price}")
                    
                    # Update lowest observed price
                    if current_price < tracking_data["lowest_price"]:
                        tracking_data["lowest_price"] = current_price
                        logger.debug(f"Updated lowest price for {symbol} #{ticket} to {current_price}")
                        
                    # Calculate new stop loss based on trailing distance
                    new_sl = tracking_data["lowest_price"] + trail_step
                    
                    # Ensure the stop loss respects minimum distance
                    min_valid_sl = current_price + min_stop_distance
                    
                    # For sell positions, stop loss must be above current price but not too close
                    # If new SL is too close to current price (below min_valid_sl), adjust it up
                    if new_sl < min_valid_sl:
                        logger.debug(f"Adjusted sell SL from {new_sl} to {min_valid_sl} to respect minimum distance ({min_stop_distance})")
                        new_sl = min_valid_sl
                    
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
                                f"from {current_sl} to {new_sl} (moved DOWN by {current_sl - new_sl:.5f})"
                            )
                            tracking_data["last_update"] = datetime.now()
                            return True
                        else:
                            logger.warning(
                                f"Failed to update trailing stop for {symbol} #{ticket} "
                                f"from {current_sl} to {new_sl}"
                            )
                    else:
                        logger.debug(
                            f"No trailing stop update needed for {symbol} #{ticket}: "
                            f"new SL {new_sl} not significantly lower than current SL {current_sl}"
                        )
                else:
                    # Periodically log that activation price hasn't been reached yet
                    last_check = tracking_data.get("last_check_time")
                    now = datetime.now()
                    if not last_check or (now - last_check).total_seconds() > 1800:  # Log once per 30 minutes
                        logger.debug(f"Waiting for activation: {symbol} #{ticket} price {current_price} needs to reach {activation_price}")
                        tracking_data["last_check_time"] = now
            
        except Exception as e:
            logger.error(f"Error applying trailing stop to position {position.get('ticket', 'unknown')}: {str(e)}")
            logger.error(traceback.format_exc())
            
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
        # Handle case when stop_loss is not set or invalid
        if stop_loss == 0.0 or abs(entry_price - stop_loss) < 0.00001:
            # If no stop loss, use a default risk percentage of entry price (1% by default)
            default_risk_pct = self.config.get("default_risk_percentage", 0.01)
            risk = entry_price * default_risk_pct
            logger.info(f"Using default risk calculation ({default_risk_pct*100:.1f}% of entry price) because stop loss is not set properly")
        else:
            risk = abs(entry_price - stop_loss)
        
        # Try to get activation factor from TRADE_EXIT_CONFIG first, then fall back to config
        trade_exit_config = self.config.get("trade_exit_config", {})
        trailing_stop_config = trade_exit_config.get("trailing_stop", {})
        
        # Use specific trailing_activation_factor if available, otherwise use activation_ratio
        activation_factor = trailing_stop_config.get("trailing_activation_factor", 
                             trailing_stop_config.get("activation_ratio", 1.0))
        
        # Apply reasonable bounds to the activation factor
        # Ensure it's between 0.2 (20% of risk) and 3.0 (300% of risk)
        activation_factor = max(0.2, min(3.0, activation_factor))
        
        logger.debug(f"Calculating activation price: direction={direction}, entry={entry_price}, " 
                    f"stop_loss={stop_loss}, risk={risk}, activation_factor={activation_factor}")
        
        # Calculate the activation price based on direction
        if direction == "buy":
            activation_price = entry_price + (risk * activation_factor)
            
            # Ensure the activation price is reasonable for market conditions
            # Maximum activation distance should be 5x the risk or 5% of entry price, whichever is smaller
            max_distance = min(risk * 5, entry_price * 0.05)
            if activation_price > entry_price + max_distance:
                activation_price = entry_price + max_distance
                logger.info(f"Adjusted activation price to be within reasonable bounds: {activation_price:.5f}")
                
            logger.debug(f"BUY activation price: {activation_price} (entry + risk*factor)")
            return activation_price
        else:
            activation_price = entry_price - (risk * activation_factor)
            
            # Ensure the activation price is reasonable for market conditions
            # Maximum activation distance should be 5x the risk or 5% of entry price, whichever is smaller
            max_distance = min(risk * 5, entry_price * 0.05)
            if activation_price < entry_price - max_distance:
                activation_price = entry_price - max_distance
                logger.info(f"Adjusted activation price to be within reasonable bounds: {activation_price:.5f}")
                
            logger.debug(f"SELL activation price: {activation_price} (entry - risk*factor)")
            return activation_price
            
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
                                price=position.get("current_price", 0.0),
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
            
            # Get latest tick data to ensure we have accurate prices
            latest_tick = self.mt5_handler.get_last_tick(symbol)
            
            # Improved tick data handling for both object and dictionary formats
            bid_price = current_price
            ask_price = current_price
            
            if latest_tick:
                try:
                    # First try object attribute access
                    if hasattr(latest_tick, 'bid') and hasattr(latest_tick, 'ask'):
                        bid_price = latest_tick.bid
                        ask_price = latest_tick.ask
                    # Then try dictionary access
                    elif isinstance(latest_tick, dict):
                        bid_price = latest_tick.get('bid', current_price)
                        ask_price = latest_tick.get('ask', current_price)
                    
                    logger.debug(f"Got latest tick for {symbol}: Bid={bid_price}, Ask={ask_price}")
                except Exception as e:
                    logger.warning(f"Error accessing tick data for {symbol}: {str(e)}. Using fallback price.")
            
            for position in positions:
                ticket = position.get("ticket", 0)
                position_type = self._get_position_type(position)
                entry_price = position.get("open_price", 0.0)
                
                # Skip invalid positions
                if not ticket:
                    logger.warning(f"Found position with no ticket for {symbol}, skipping")
                    continue
                
                # Use appropriate price based on position type
                market_price = ask_price if position_type == "buy" else bid_price
                
                # Update position with current market price
                position["current_price"] = market_price
                
                # Add detailed logging
                logger.info(f"Position #{ticket}: Type={position_type}, Entry={entry_price}, Current={market_price}")
                
                # Update the position in our tracking
                if ticket in self.active_trades:
                    # Update current price in our tracked position
                    self.active_trades[ticket]["current_price"] = market_price
                    logger.debug(f"Updated tracked position #{ticket} with current price {market_price}")
                    
                    # Calculate current profit
                    volume = position.get("volume", 0.0)
                    profit_pips = (market_price - entry_price) * 10000 if position_type == "buy" else (entry_price - market_price) * 10000
                    logger.debug(f"Position #{ticket} current P/L: {profit_pips:.1f} pips")
                    
                    # Apply trailing stop logic if enabled
                    if self.trailing_stop_enabled and ticket in self.trailing_stop_data:
                        trailing_result = self._apply_trailing_stop(position)
                        if trailing_result:
                            logger.info(f"Successfully applied trailing stop to position #{ticket}")
                        else:
                            logger.debug(f"Applied trailing stop to position #{ticket} (no change needed)")
                else:
                    # Position not in our tracking, add it
                    logger.info(f"Adding position #{ticket} to tracking")
                    self.active_trades[ticket] = position
                
        except Exception as e:
            logger.error(f"Error updating positions for {symbol}: {str(e)}")
            logger.error(traceback.format_exc()) 