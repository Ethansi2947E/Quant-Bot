import traceback
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import time
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np

from src.telegram.telegram_bot import TelegramBot
from src.mt5_handler import MT5Handler
from src.utils.market_utils import calculate_pip_value, convert_pips_to_price, convert_price_to_pips
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
                            if isinstance(tick, dict):
                                current_price = tick.get('ask', 0.0) if position_type == "buy" else tick.get('bid', 0.0)
                            else:
                                current_price = getattr(tick, 'ask', 0.0) if position_type == "buy" else getattr(tick, 'bid', 0.0)
                            
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
            
            # Get the trailing step size from base config
            from config.config import TRADE_EXIT_CONFIG
            base_config = TRADE_EXIT_CONFIG.get('trailing_stop', {})
            
            # Get instrument-specific configuration if adaptive mode is enabled
            if base_config.get('mode', 'pips') == 'adaptive':
                instrument_config = self._get_instrument_config(symbol)
                logger.info(f"Using adaptive config for {symbol}: {instrument_config}")
                
                # Use instrument-specific parameters, fallback to base config
                trailing_mode = instrument_config.get('mode', base_config.get('mode', 'pips'))
                trail_step_pips = instrument_config.get('trail_points', base_config.get('trail_points', 15.0))
                atr_multiplier = instrument_config.get('atr_multiplier', base_config.get('atr_multiplier', 2.0))
                atr_period = instrument_config.get('atr_period', base_config.get('atr_period', 14))
                percent = instrument_config.get('percent', base_config.get('percent', 0.008))
                break_even_pips = instrument_config.get('break_even_pips', base_config.get('break_even_pips', 5))
            else:
                # Use base configuration only
                trailing_mode = base_config.get('mode', 'pips')
                trail_step_pips = base_config.get('trail_points', base_config.get('trail_step_pips', 15.0))
                atr_multiplier = base_config.get('atr_multiplier', 2.0)
                atr_period = base_config.get('atr_period', 14)
                percent = base_config.get('percent', 0.008)
                break_even_pips = base_config.get('break_even_pips', 5)
            
            # Get break even config (always from base config for consistency)
            break_even_enabled = base_config.get('break_even_enabled', True)
            break_even_buffer_pips = base_config.get('break_even_buffer_pips', 0.5)

            # Get pip value for this symbol using the utility function
            pip_value = calculate_pip_value(symbol, mt5_handler=self.mt5_handler)
            
            # Calculate break-even thresholds and buffers (always needed)
            break_even_threshold = break_even_pips * pip_value
            break_even_buffer = break_even_buffer_pips * pip_value

            # Trailing stop mode selection
            atr = None
            if trailing_mode == 'atr':
                try:
                    from src.utils.indicators import calculate_atr
                    # Use minimum of 50 bars or 3x ATR period for calculation
                    required_bars = max(50, atr_period * 3)
                    df = self.mt5_handler.get_market_data(symbol, 'M1', required_bars)
                    if df is not None and len(df) >= atr_period:
                        atr_series = calculate_atr(df, atr_period)
                        if isinstance(atr_series, pd.Series):
                            atr = float(atr_series.iloc[-1])
                        elif isinstance(atr_series, (np.ndarray, list)):
                            atr = float(atr_series[-1])
                        elif isinstance(atr_series, float):
                            atr = atr_series
                        elif isinstance(atr_series, int):
                            atr = float(atr_series)
                        elif isinstance(atr_series, pd.DataFrame):
                            logger.warning("ATR series is a DataFrame, cannot extract ATR value.")
                            atr = None
                        else:
                            logger.warning(f"ATR series is of unsupported type: {type(atr_series)}")
                            atr = None
                except Exception as e:
                    logger.warning(f"Failed to calculate ATR for {symbol}: {e}")
            if trailing_mode == 'atr' and atr is not None:
                trailing_distance = atr * atr_multiplier
                logger.info(f"Using ATR-based trailing stop for {symbol}: ATR={atr:.5f}, multiplier={atr_multiplier}, trailing_distance={trailing_distance:.5f}")
            elif trailing_mode == 'percent':
                trailing_distance = current_price * percent
                logger.info(f"Using percent-based trailing stop for {symbol}: {percent*100:.2f}%, trailing_distance={trailing_distance:.5f}")
            else:
                trailing_distance = trail_step_pips * pip_value
                logger.info(f"Using pip-based trailing stop for {symbol}: {trail_step_pips} pips, trailing_distance={trailing_distance:.5f}")

            # Log the selected parameters for transparency
            logger.info(f"Trailing stop config for {symbol}: mode={trailing_mode}, trail_pips={trail_step_pips}, "
                       f"atr_mult={atr_multiplier}, percent={percent*100:.1f}%, break_even={break_even_pips} pips")

            # Get minimum stop level from MT5
            min_stop_distance = self.mt5_handler.get_min_stop_distance(symbol)
            
            logger.debug(f"Minimum stop distance for {symbol}: {min_stop_distance}")
            
            # --- Break Even Logic ---
            if break_even_enabled:
                if position_type == "buy":
                    profit = current_price - entry_price
                    # Only move SL to break even if profit threshold reached and SL is below entry
                    if profit >= break_even_threshold and current_sl < entry_price:
                        new_sl = entry_price + break_even_buffer
                        # Ensure new SL is not too close to current price
                        min_valid_sl = current_price - min_stop_distance
                        if new_sl > min_valid_sl:
                            logger.debug(f"Adjusted break even SL from {new_sl} to {min_valid_sl} to respect minimum distance ({min_stop_distance})")
                            new_sl = min_valid_sl
                        if new_sl > current_sl + (0.1 * pip_value):
                            success = self.mt5_handler.modify_position(
                                ticket=ticket,
                                new_sl=new_sl,
                                new_tp=position.get("tp", 0.0)
                            )
                            if success:
                                logger.info(f"Moved SL to break even for BUY {symbol} #{ticket}: {current_sl} -> {new_sl}")
                                return True
                            else:
                                logger.warning(f"Failed to move SL to break even for BUY {symbol} #{ticket} from {current_sl} to {new_sl}")
                elif position_type == "sell":
                    profit = entry_price - current_price
                    if profit >= break_even_threshold and (current_sl > entry_price or current_sl == 0.0):
                        new_sl = entry_price - break_even_buffer
                        min_valid_sl = current_price + min_stop_distance
                        if new_sl < min_valid_sl:
                            logger.debug(f"Adjusted break even SL from {new_sl} to {min_valid_sl} to respect minimum distance ({min_stop_distance})")
                            new_sl = min_valid_sl
                        if new_sl < current_sl - (0.1 * pip_value) or current_sl == 0.0:
                            success = self.mt5_handler.modify_position(
                                ticket=ticket,
                                new_sl=new_sl,
                                new_tp=position.get("tp", 0.0)
                            )
                            if success:
                                logger.info(f"Moved SL to break even for SELL {symbol} #{ticket}: {current_sl} -> {new_sl}")
                                return True
                            else:
                                logger.warning(f"Failed to move SL to break even for SELL {symbol} #{ticket} from {current_sl} to {new_sl}")
            
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
                    new_sl = tracking_data["highest_price"] - trailing_distance
                    
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
                    new_sl = tracking_data["lowest_price"] + trailing_distance
                    
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
            if isinstance(latest_tick, dict):
                bid_price = latest_tick.get('bid', current_price)
                ask_price = latest_tick.get('ask', current_price)
            else:
                bid_price = getattr(latest_tick, 'bid', current_price)
                ask_price = getattr(latest_tick, 'ask', current_price)
            
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
                    if position_type == "buy":
                        profit_pips = convert_price_to_pips(market_price - entry_price, symbol, mt5_handler=self.mt5_handler)
                    else:
                        profit_pips = convert_price_to_pips(entry_price - market_price, symbol, mt5_handler=self.mt5_handler)
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

    def _get_instrument_config(self, symbol: str) -> Dict[str, Any]:
        """
        Automatically detect instrument type and return appropriate trailing stop configuration.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'Volatility 10 Index')
            
        Returns:
            Dictionary containing instrument-specific trailing stop parameters
        """
        from config.config import TRADE_EXIT_CONFIG
        
        # Get instrument-specific configurations
        instrument_configs = TRADE_EXIT_CONFIG.get('trailing_stop', {}).get('instrument_configs', {})
        
        # Check each instrument category
        for category, config in instrument_configs.items():
            if symbol in config.get('symbols', []):
                logger.info(f"Detected {symbol} as {category} - using specialized config")
                return config
        
        # If no exact match, try pattern matching for better detection
        symbol_upper = symbol.upper()
        
        # Forex major pairs pattern matching
        forex_majors = ['EUR', 'GBP', 'USD', 'JPY', 'CAD', 'AUD', 'NZD', 'CHF']
        if len(symbol) == 6 and symbol_upper[:3] in forex_majors and symbol_upper[3:] in forex_majors:
            logger.info(f"Pattern-detected {symbol} as forex_major")
            return instrument_configs.get('forex_major', {})
        
        # Volatility indices pattern matching
        if 'volatility' in symbol_upper and 'index' in symbol_upper:
            logger.info(f"Pattern-detected {symbol} as volatility_indices")
            return instrument_configs.get('volatility_indices', {})
        
        # Crash/Boom indices pattern matching
        if ('crash' in symbol_upper or 'boom' in symbol_upper) and 'index' in symbol_upper:
            logger.info(f"Pattern-detected {symbol} as crash_boom_indices")
            return instrument_configs.get('crash_boom_indices', {})
        
        # Jump indices pattern matching
        if 'jump' in symbol_upper and 'index' in symbol_upper:
            logger.info(f"Pattern-detected {symbol} as jump_indices")
            return instrument_configs.get('jump_indices', {})
        
        # Step/Range indices pattern matching
        if ('step' in symbol_upper or 'range' in symbol_upper) and 'index' in symbol_upper:
            logger.info(f"Pattern-detected {symbol} as step_range_indices")
            return instrument_configs.get('step_range_indices', {})
        
        # Metals pattern matching
        if symbol_upper.startswith('XAU') or symbol_upper.startswith('XAG') or 'gold' in symbol_upper or 'silver' in symbol_upper:
            logger.info(f"Pattern-detected {symbol} as metals")
            return instrument_configs.get('metals', {})
        
        # Crypto pattern matching
        if 'btc' in symbol_upper or 'eth' in symbol_upper or 'bitcoin' in symbol_upper or 'ethereum' in symbol_upper:
            logger.info(f"Pattern-detected {symbol} as crypto")
            return instrument_configs.get('crypto', {})
        
        # Default to forex_major if no pattern matches (safest default)
        logger.warning(f"Could not detect instrument type for {symbol}, using forex_major defaults")
        return instrument_configs.get('forex_major', {})

    def test_adaptive_detection(self) -> None:
        """
        Test the adaptive instrument detection system.
        This method demonstrates how different symbols are automatically detected and configured.
        """
        test_symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY',  # Forex majors
            'Volatility 10 Index', 'Volatility 25 Index',  # Volatility indices
            'Crash 500 Index', 'Boom 1000 Index',  # Crash/Boom indices
            'XAUUSD', 'BTCUSD',  # Metals and crypto
            'Jump 50 Index', 'Step Index',  # Jump and step indices
            'UNKNOWN_SYMBOL'  # Unknown symbol
        ]
        
        logger.info("=== Testing Adaptive Instrument Detection ===")
        
        for symbol in test_symbols:
            config = self._get_instrument_config(symbol)
            logger.info(f"{symbol:20} -> Mode: {config.get('mode', 'N/A'):8} | "
                       f"Trail: {config.get('trail_points', 0):6.1f} pips | "
                       f"ATR: {config.get('atr_multiplier', 0):4.1f}x | "
                       f"Percent: {config.get('percent', 0)*100:5.1f}% | "
                       f"BE: {config.get('break_even_pips', 0):3.0f} pips")
        
        logger.info("=== End Adaptive Detection Test ===") 