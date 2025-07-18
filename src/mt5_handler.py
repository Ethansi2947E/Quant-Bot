# mypy: ignore-errors
# pyright: reportAttributeAccessIssue=false
# flake8: noqa

import MetaTrader5 as mt5  # type: ignore
from datetime import datetime, timedelta, UTC
import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any
import time
import traceback
import re 
import math 
import asyncio
import json

from config.config import MT5_CONFIG, TRADING_CONFIG, RISK_MANAGER_CONFIG
import typing
if typing.TYPE_CHECKING:
    from src.risk_manager import RiskManager  # For type hints only, no circular import

# Define MetaTrader5 attributes for type checking
# This tells the type checker that these methods exist on the mt5 module
if False:  # This block is never executed, just for type checking
    def __mt5_type_hints():
        # Add MetaTrader5 methods to help type checkers
        # Use type: ignore comments to suppress type errors
        mt5.shutdown()  # type: ignore
        mt5.initialize()  # type: ignore
        mt5.login()  # type: ignore
        mt5.last_error()  # type: ignore
        mt5.account_info()  # type: ignore
        mt5.symbol_select()  # type: ignore
        mt5.copy_rates_from_pos()  # type: ignore
        mt5.symbol_info()  # type: ignore
        mt5.symbol_info_tick()  # type: ignore
        mt5.positions_get()  # type: ignore
        mt5.order_send()  # type: ignore
        mt5.history_orders_get()  # type: ignore
        mt5.history_deals_get()  # type: ignore
        mt5.order_calc_margin()  # type: ignore
        mt5.is_connected = True  # type: ignore
        mt5.terminal_info()  # type: ignore

# Import RiskManager
from src.risk_manager import RiskManager

# Singleton instance for global reference
_mt5_handler_instance = None

class MT5Handler:
    def __init__(self):
        self.connected = False
        self.initialize()
        self._last_error = None  # Add error tracking
        # Initialize risk manager reference
        self.risk_manager = None

    @classmethod
    def get_instance(cls):
        """Return the singleton instance, creating it if it doesn't exist."""
        global _mt5_handler_instance
        if _mt5_handler_instance is None:
            _mt5_handler_instance = cls()
        return _mt5_handler_instance

    def initialize(self) -> bool:
        """Initialize connection to MT5 terminal."""
        try:
            # First, try to shut down any existing MT5 connections
            try:
                if hasattr(mt5, 'shutdown'):
                    mt5.shutdown()  # type: ignore
                logger.debug("Cleaned up any existing MT5 connections before initialization")
            except Exception as e:
                logger.debug(f"MT5 shutdown during initialization: {str(e)}")
                # Continue despite shutdown errors

            # Initialize MT5
            if not hasattr(mt5, 'initialize'):
                logger.error("MetaTrader5 module does not have 'initialize' method")
                return False

            if not mt5.initialize():  # type: ignore
                logger.error("MT5 initialization failed")
                return False

            # Get MT5_CONFIG from config to ensure we have the latest values
            login = MT5_CONFIG["login"]
            server = MT5_CONFIG["server"]
            password = MT5_CONFIG["password"]
            timeout = MT5_CONFIG.get("timeout", 60000)

            logger.debug(f"Logging in to MT5 with server: {server}, login: {login}")

            # Login to MT5
            if not mt5.login(  # type: ignore
                login=login,
                server=server,
                password=password,
                timeout=timeout
            ):
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}")  # type: ignore
                return False

            self.connected = True
            logger.info("MT5 connection established successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            logger.error("MT5 not connected")
            return {}

        try:
            # First check if MT5 is still connected
            if not self.is_connected():
                logger.error("MT5 connection lost when trying to get account info")
                return {}

            account_info = mt5.account_info()  # type: ignore
            if account_info is None:
                error = "Unknown error"
                if hasattr(mt5, 'last_error'):
                    error = str(mt5.last_error())
                logger.error(f"Failed to get account info. Error: {error}")
                return {}

            # Create detailed account info dictionary
            result = {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "leverage": account_info.leverage,
                "currency": account_info.currency,
                "login": account_info.login,
                "name": account_info.name,
                "server": account_info.server,
                "profit": account_info.profit,
                "margin_level": account_info.margin_level
            }

            return result
        except Exception as e:
            logger.error(f"Exception in get_account_info: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def get_latest_candle_time(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        Get the timestamp of the most recent candle for a symbol and timeframe.
        This is a lightweight check to see if a new candle has formed.
        
        Args:
            symbol: Symbol to check
            timeframe: Timeframe to check
            
        Returns:
            The timestamp (as an integer Unix epoch) of the latest candle, or None if error.
        """
        if not self.connected:
            return None

        try:
            mt5_timeframe = self._get_mt5_timeframe(timeframe)
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)
            
            if rates is None or len(rates) == 0:
                return None
                
            return int(rates[0]['time'])
            
        except Exception as e:
            logger.trace(f"Could not get latest candle time for {symbol}/{timeframe}: {e}")
            return None

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        num_candles: int = 1000
    ) -> Optional[pd.DataFrame]:
        logger.debug(f"[MT5Handler] get_market_data: symbol={symbol}, timeframe={timeframe}, num_candles={num_candles}")
        """
        Get market data for a symbol and timeframe.

        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            num_candles: Number of candles to get

        Returns:
            DataFrame with market data or None if error
        """
        # Try to ensure connection first
        if not self.connected:
            if not self.initialize():
                logger.error("MT5 not connected and failed to reconnect")
                return None

        # Keep track of connection recovery attempts
        recovery_attempts = 0
        max_recovery_attempts = 3

        while recovery_attempts <= max_recovery_attempts:
            try:
                # Map timeframe string to MT5 timeframe constant
                mt5_timeframe = self._get_mt5_timeframe(timeframe)

                # Select symbol first
                if not mt5.symbol_select(symbol, True):  # type: ignore
                    error_code = mt5.last_error()  # type: ignore
                    # Check if it's a connection error
                    if error_code[0] == -10004:  # No IPC connection
                        recovery_attempts += 1
                        if recovery_attempts <= max_recovery_attempts:
                            logger.warning(f"MT5 connection lost while selecting symbol {symbol}, attempting to reconnect (attempt {recovery_attempts}/{max_recovery_attempts})")
                            if self.initialize():
                                logger.info("MT5 connection re-established, retrying data fetch")
                                continue
                            else:
                                time.sleep(1)  # Brief pause before retry
                                continue
                    # Don't log as error for symbol not found - just as warning
                    elif error_code[0] == 4301:  # Symbol not found
                        logger.warning(f"Symbol {symbol} not found in MT5. Please check if this symbol is available in your broker's market watch.")
                        return None
                    else:
                        logger.error(f"Failed to select symbol {symbol} for data retrieval. This symbol may not be available in your MT5 account.")
                        logger.error(f"MT5 error: {error_code}")
                        return None

                # Get rates
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)  # type: ignore

                # Check for connection errors
                if rates is None:
                    error_code = mt5.last_error()  # type: ignore
                    if error_code[0] == -10004:  # No IPC connection
                        recovery_attempts += 1
                        if recovery_attempts <= max_recovery_attempts:
                            logger.warning(f"MT5 connection lost while getting rates for {symbol}, attempting to reconnect (attempt {recovery_attempts}/{max_recovery_attempts})")
                            if self.initialize():
                                logger.info("MT5 connection re-established, retrying data fetch")
                                continue
                            else:
                                time.sleep(1)  # Brief pause before retry
                                continue
                        else:
                            logger.error(f"Failed to recover MT5 connection after {max_recovery_attempts} attempts")
                            return None
                    else:
                        logger.warning(f"No data returned for {symbol} on {timeframe}")
                        return None

                if len(rates) == 0:
                    logger.warning(f"Empty data set returned for {symbol} on {timeframe}")
                    return None

                # Convert to DataFrame
                df = pd.DataFrame(rates)

                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'], unit='s')

                # Rename columns to match our convention
                df.rename(columns={
                    'time': 'datetime',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'tick_volume': 'volume',
                    'spread': 'spread',
                    'real_volume': 'real_volume'
                }, inplace=True)

                # Set datetime as index
                df.set_index('datetime', inplace=True)

                return df

            except Exception as e:
                recovery_attempts += 1
                self._last_error = str(e)

                # Check if it looks like a connection error
                if "IPC" in str(e) or "connection" in str(e).lower():
                    if recovery_attempts <= max_recovery_attempts:
                        logger.warning(f"Connection error for {symbol} on {timeframe}, attempting to reconnect (attempt {recovery_attempts}/{max_recovery_attempts}): {str(e)}")
                        if self.initialize():
                            logger.info("MT5 connection re-established, retrying data fetch")
                            continue
                        time.sleep(1)  # Brief pause before retry
                    else:
                        logger.error(f"Failed to recover connection after {max_recovery_attempts} attempts")
                else:
                    logger.error(f"Error getting market data for {symbol} on {timeframe}: {str(e)}")

                if recovery_attempts > max_recovery_attempts:
                    return None

        # If we get here, all recovery attempts failed
        return None

    def _adjust_volume_to_broker_limits(self, symbol: str, requested_volume: float) -> float:
        """
        Adjust the requested lot size to comply with broker's min/max/step constraints for the symbol.
        Args:
            symbol (str): Trading symbol
            requested_volume (float): Requested lot size
        Returns:
            float: Adjusted lot size within allowed limits
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"Could not get symbol info for {symbol}, using requested volume {requested_volume}")
            return requested_volume
        min_vol = getattr(symbol_info, 'volume_min', 0.01)
        max_vol = getattr(symbol_info, 'volume_max', 100.0)
        step = getattr(symbol_info, 'volume_step', 0.01)
        # Clamp to min/max
        adjusted = max(min(requested_volume, max_vol), min_vol)
        # Align to step
        adjusted = round(adjusted / step) * step
        # Ensure precision (usually 2 decimals)
        adjusted = round(adjusted, 2)
        if adjusted != requested_volume:
            logger.info(f"Adjusted requested volume {requested_volume} to {adjusted} for {symbol} (min={min_vol}, max={max_vol}, step={step})")
        return adjusted

    def place_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: float,
        take_profit: Optional[float] = None, # Can now be None
        comment: str = "",
        take_profits: Optional[List[float]] = None # New optional list of TPs
    ) -> Optional[Dict[str, Any]]:
        """
        Place a market order (BUY or SELL).
        If `take_profits` list is provided, its first element is used as the TP.
        Otherwise, the single `take_profit` value is used.
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected")
                return None

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None

            # Check if symbol is visible in MarketWatch
            if not symbol_info.visible:
                logger.warning(f"{symbol} not visible, trying to add it")
                if not mt5.symbol_select(symbol, True):  # type: ignore
                    logger.error(f"Failed to add {symbol} to MarketWatch")
                    return None

            # Get account info for logging
            account_info = mt5.account_info()  # type: ignore
            if account_info:
                logger.info(f"Account balance: {account_info.balance:.2f}, Free margin: {account_info.margin_free:.2f}, Equity: {account_info.equity:.2f}")

            # Log trading state
            logger.debug(f"Symbol {symbol} trading state: Trade Mode={symbol_info.trade_mode}, Visible={symbol_info.visible}")

            # Map order type to MT5 constant, making it case-insensitive
            action_str = order_type.upper()
            if action_str == "BUY":
                action = mt5.ORDER_TYPE_BUY
                price = symbol_info.ask
            elif action_str == "SELL":
                action = mt5.ORDER_TYPE_SELL
                price = symbol_info.bid
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None

            # Log current prices
            logger.debug(f"Current prices for {symbol}: Ask={symbol_info.ask}, Bid={symbol_info.bid}, Spread={symbol_info.ask - symbol_info.bid}")

            # --- ENFORCE TRUE MIN STOP DISTANCE ---
            # Use get_min_stop_distance to get the most accurate min stop distance
            min_stop_distance_mt5 = symbol_info.point * symbol_info.trade_stops_level
            min_stop_distance_fallback = self.get_min_stop_distance(symbol)
            min_stop_distance = max(min_stop_distance_mt5, min_stop_distance_fallback)
            logger.debug(f"[Order] {symbol}: min_stop_distance_mt5={min_stop_distance_mt5}, min_stop_distance_fallback={min_stop_distance_fallback}, used={min_stop_distance}")

            # Adjust stop loss and take profit if they're too close to the entry price
            if action == mt5.ORDER_TYPE_BUY:
                if stop_loss is not None and stop_loss != 0 and stop_loss >= price - min_stop_distance:
                    logger.warning(f"Stop loss for BUY order too close to entry, adjusting: SL ({stop_loss}) → ({price - min_stop_distance})")
                    stop_loss = price - min_stop_distance
                if take_profit is not None and take_profit != 0 and take_profit <= price + min_stop_distance:
                    logger.warning(f"Take profit for BUY order too close to entry, adjusting: TP ({take_profit}) → ({price + min_stop_distance})")
                    take_profit = price + min_stop_distance
            else:  # SELL
                if stop_loss is not None and stop_loss != 0 and stop_loss <= price + min_stop_distance:
                    logger.warning(f"Stop loss for SELL order too close to entry, adjusting: SL ({stop_loss}) → ({price + min_stop_distance})")
                    stop_loss = price + min_stop_distance
                if take_profit is not None and take_profit != 0 and take_profit >= price - min_stop_distance:
                    logger.warning(f"Take profit for SELL order too close to entry, adjusting: TP ({take_profit}) → ({price - min_stop_distance})")
                    take_profit = price - min_stop_distance

            # --- NEW: Round price and SL/TP to the correct number of digits allowed by the symbol ---
            try:
                digits = int(getattr(symbol_info, "digits", 2))  # default to 2 if attribute missing
                price = round(price, digits)
                if stop_loss is not None and stop_loss != 0:
                    stop_loss = round(stop_loss, digits)
                if take_profit is not None and take_profit != 0:
                    take_profit = round(take_profit, digits)
                logger.debug(f"Rounded prices to {digits} digits → Price: {price}, SL: {stop_loss}, TP: {take_profit}")
            except Exception as e:
                logger.warning(f"Failed to round prices for {symbol}: {e}")
            # ------------------------------------------------------------------------------

            # Log original volume request
            logger.info(f"Requested position size: {volume:.4f} lots")

            # --- NEW: Adjust volume to broker limits (min/max/step) ---
            adjusted_volume = self._adjust_volume_to_broker_limits(symbol, volume)
            if adjusted_volume != volume:
                logger.info(f"Volume adjusted to broker limits: {adjusted_volume:.4f} lots (was {volume:.4f})")

            # --- Optionally, also adjust for margin constraints using risk manager as before ---
            if self.risk_manager:
                try:
                    if account_info and 'balance' in account_info:
                        account_balance = account_info['balance']
                        stop_loss_price = stop_loss if stop_loss != 0 else price * 0.95
                        if hasattr(self.risk_manager, 'calculate_position_size'):
                            risk_percent = self.risk_manager.max_risk_per_trade * 100 if hasattr(self.risk_manager, 'max_risk_per_trade') else 1.0
                            margin_adjusted = self.risk_manager.calculate_position_size(
                                account_balance=account_balance,
                                risk_per_trade=risk_percent,
                                entry_price=price,
                                stop_loss_price=stop_loss_price,
                                symbol=symbol
                            )
                            # Take the minimum of broker-limited and margin-limited size
                            if margin_adjusted < adjusted_volume:
                                logger.info(f"Further adjusted position size to {margin_adjusted:.4f} lots due to margin constraints")
                            adjusted_volume = min(adjusted_volume, margin_adjusted)
                except Exception as e:
                    logger.warning(f"Failed to adjust position size for margin: {str(e)}. Using broker-limited size: {adjusted_volume:.4f} lots")

            # Final check: ensure volume is still within broker limits after all adjustments
            adjusted_volume = self._adjust_volume_to_broker_limits(symbol, adjusted_volume)

            if adjusted_volume == 0:
                logger.error(f"Cannot place order: No margin or volume available for {symbol}")
                return None

            # Get appropriate filling mode for this symbol
            filling_mode = self.get_symbol_filling_mode(symbol)
            logger.debug(f"Using filling mode {filling_mode} for {symbol}")

            # Ensure comment is valid
            if comment is None or not isinstance(comment, str):
                comment = f"MT5Bot-{symbol}"
            # Limit comment length to 31 characters to comply with MT5 requirements
            comment = comment[:31]

            # Remove any potentially problematic characters from comment
            comment = re.sub(r'[^a-zA-Z0-9_\-\.]', '', comment)

            # If comment is empty after sanitization, use a default
            if not comment:
                comment = f"MT5Bot{symbol.replace(' ', '')}"

            # --- Multi-TP Handling ---
            final_tp = 0.0
            if take_profits and isinstance(take_profits, list) and len(take_profits) > 0:
                final_tp = take_profits[0]
                logger.info(f"Using first TP ({final_tp}) from the provided list for order placement.")
            elif take_profit is not None:
                final_tp = take_profit
            
            # Prepare the request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": adjusted_volume,
                "type": action,  # Use MT5 constant instead of string
                "price": price,
                "sl": stop_loss if stop_loss != 0 else None,
                "tp": final_tp if final_tp != 0 else None,  # Use the determined final_tp
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            if stop_loss == 0:
                logger.debug(f"Converting SL value of 0 to None for {symbol} {order_type} order")
            if final_tp == 0:
                logger.debug(f"Converting TP value of 0 to None for {symbol} {order_type} order")

            # Try multiple times with increasing deviation and different filling modes if needed
            max_retries = 3
            filling_modes_to_try = [filling_mode]

            # Prepare fallback filling modes in case primary one fails
            if filling_mode != mt5.ORDER_FILLING_IOC:
                filling_modes_to_try.append(mt5.ORDER_FILLING_IOC)
            if filling_mode != mt5.ORDER_FILLING_FOK and mt5.ORDER_FILLING_FOK not in filling_modes_to_try:
                filling_modes_to_try.append(mt5.ORDER_FILLING_FOK)

            for filling_mode in filling_modes_to_try:
                request["type_filling"] = filling_mode
                logger.debug(f"Trying with filling mode: {filling_mode}")

                for attempt in range(max_retries):
                    result = mt5.order_send(request)  # type: ignore
                    if result is None:
                        error_info = mt5.last_error()  # type: ignore
                        logger.error(f"Failed to send order: {error_info}")

                        if "unsupported filling" in str(error_info).lower():
                            # Break this attempt loop and try next filling mode
                            logger.warning(f"Unsupported filling mode {filling_mode}, will try alternative")
                            break

                        continue

                    logger.debug(f"Order result: {json.dumps(result._asdict(), default=str)}")

                    # --- START REPLACEMENT BLOCK ---

                    if result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 10009:  # Success codes
                        logger.info(f"Order executed successfully: Ticket {result.order}")
                        return {
                            "ticket": result.order,
                            "volume": result.volume,
                            "price": result.price,
                            "comment": comment
                        }

                    # --- Revised handling for retcode 10030 ---
                    # 10030 can signal either an unsupported filling mode _or_ a broker-imposed position/volume limit.
                    # We inspect the comment field to distinguish the cases.
                    if result.retcode == 10030:
                        comment_lower = str(getattr(result, "comment", "")).lower()
                        if "unsupported filling" in comment_lower:
                            logger.warning(
                                f"Unsupported filling mode {filling_mode} detected via retcode 10030. "
                                "Trying alternative filling mode."
                            )
                            break  # break out of attempts loop to try the next filling_mode
                        else:
                            logger.error(
                                "Order failed with retcode 10030 (likely limit on positions/volume): "
                                f"{getattr(result, 'comment', '')}"
                            )
                            return None  # Fail immediately, do not retry
                    
                    if result.retcode == mt5.TRADE_RETCODE_MARKET_CLOSED:
                        logger.error(f"Order failed: Market is closed for {symbol}.")
                        return None # Fail immediately

                    # Handle retryable errors
                    if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                        logger.warning(f"Requote on attempt {attempt + 1}. Updating price and retrying.")
                        tick = mt5.symbol_info_tick(symbol)
                        if tick:
                            request["price"] = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
                        request["deviation"] += 10 # Increase slippage tolerance
                        continue # Continue to next attempt in the loop

                    if result.retcode == 10016:
                        comment_lower = str(getattr(result, "comment", "")).lower()

                        # Case 1: Invalid stops (distance too small)
                        if "invalid stops" in comment_lower or "invalid sl" in comment_lower:
                            # Widen SL/TP distances and retry
                            buffer = min_stop_distance  # reuse previously-calculated min distance

                            if action == mt5.ORDER_TYPE_BUY:
                                new_sl = price - (buffer * 2)
                                new_tp = price + (buffer * 2)
                            else:  # SELL
                                new_sl = price + (buffer * 2)
                                new_tp = price - (buffer * 2)

                            # Round to correct precision
                            digits = int(getattr(symbol_info, "digits", 2))
                            new_sl = round(new_sl, digits)
                            new_tp = round(new_tp, digits)

                            logger.warning(
                                f"Invalid stops detected on attempt {attempt + 1}. "
                                f"Adjusting SL to {new_sl}, TP to {new_tp} and retrying."
                            )

                            request["sl"] = new_sl if new_sl != 0 else None
                            request["tp"] = new_tp if new_tp != 0 else None
                            # Also bump deviation slightly to reduce rejections
                            request["deviation"] += 5
                            continue  # retry with the adjusted stops

                        # Case 2: generic dealer cancellation – just increase deviation and retry
                        logger.warning(
                            f"Order canceled by dealer on attempt {attempt + 1}. Increasing deviation and retrying."
                        )
                        request["deviation"] += 15
                        continue # Continue to next attempt

                    # Fallback for other errors
                    error_description = self.get_error_info(result.retcode)
                    logger.error(f"Order failed with unhandled error code {result.retcode}: {error_description}")
                    
                    # If the comment suggests a filling mode issue, break to try the next mode
                    if "unsupported filling" in str(result.comment).lower():
                        logger.warning("Unsupported filling mode detected by comment, trying next mode.")
                        break # Break from attempts loop to try next filling mode

                    # For other errors, just retry with more deviation
                    request["deviation"] += 10
                
                # --- END REPLACEMENT BLOCK ---

            logger.error("Failed to place order after trying all filling modes and retries")
            try:
                # Capture the current call-stack for deeper diagnostics
                stack_str = "\n".join(traceback.format_stack())
                logger.debug(f"Call-stack at order failure for {symbol} {order_type}:\n{stack_str}")
            except Exception as _stack_err:
                logger.warning(f"Could not capture failure stack trace: {_stack_err}")

            # Log the last error coming from MT5 (if any)
            try:
                mt5_err = mt5.last_error()  # type: ignore
                logger.error(f"MT5 last_error after failure: {mt5_err}")
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"An unexpected exception occurred in place_market_order for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    def place_limit_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
        comment: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Place a limit order (BUY_LIMIT or SELL_LIMIT)."""
        if not self.connected:
            logger.error("MT5 not connected")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None

        if not symbol_info.visible:
            logger.warning(f"{symbol} not visible, trying to add it")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to add {symbol} to MarketWatch")
                return None

        if order_type == "BUY":
            action = mt5.ORDER_TYPE_BUY_LIMIT
        elif order_type == "SELL":
            action = mt5.ORDER_TYPE_SELL_LIMIT
        else:
            logger.error(f"Invalid order type for limit order: {order_type}")
            return None

        try:
            digits = int(getattr(symbol_info, "digits", 2))
            limit_price = round(limit_price, digits)
            if stop_loss != 0:
                stop_loss = round(stop_loss, digits)
            if take_profit != 0:
                take_profit = round(take_profit, digits)
            logger.debug(f"Rounded prices to {digits} digits -> Price: {limit_price}, SL: {stop_loss}, TP: {take_profit}")
        except Exception as e:
            logger.warning(f"Failed to round prices for {symbol}: {e}")

        adjusted_volume = self._adjust_volume_to_broker_limits(symbol, volume)
        if adjusted_volume == 0:
            logger.error(f"Cannot place order: volume is zero after adjustments for {symbol}")
            return None

        filling_mode = self.get_symbol_filling_mode(symbol)

        if comment is None or not isinstance(comment, str):
            comment = f"MT5Bot-{symbol}"
        comment = comment[:31]
        comment = re.sub(r'[^a-zA-Z0-9_\\-\\.]', '', comment)
        if not comment:
            comment = f"MT5Bot{symbol.replace(' ', '')}"

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": adjusted_volume,
            "type": action,
            "price": limit_price,
            "sl": stop_loss if stop_loss != 0 else 0.0,
            "tp": take_profit if take_profit != 0 else 0.0,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)

        if result is None:
            error_info = mt5.last_error()
            logger.error(f"Failed to send limit order: {error_info}")
            return None

        logger.debug(f"Limit order result: {json.dumps(result._asdict(), default=str)}")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Limit order placed successfully: Ticket {result.order}")
            return {
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": comment
            }
        else:
            error_description = self.get_error_info(result.retcode)
            logger.error(f"Limit order failed with error code {result.retcode}: {error_description}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        if not self.connected:
            logger.error("MT5 not connected")
            return []

        # Add recovery mechanism for connection issues
        recovery_attempts = 0
        max_recovery_attempts = 3

        while recovery_attempts <= max_recovery_attempts:
            try:
                positions = mt5.positions_get()  # type: ignore
                if positions is None:
                    error = mt5.last_error()  # type: ignore
                    # Check if it's a connection error
                    if error[0] == -10004:  # No IPC connection
                        recovery_attempts += 1
                        if recovery_attempts <= max_recovery_attempts:
                            logger.warning(f"MT5 connection lost when getting positions, attempting to reconnect (attempt {recovery_attempts}/{max_recovery_attempts})")
                            if self.initialize():
                                logger.info("MT5 connection re-established, retrying positions fetch")
                                continue
                            else:
                                time.sleep(1)  # Brief pause before retry
                                continue
                        else:
                            logger.error("Failed to recover MT5 connection after multiple attempts")
                            return []
                    else:
                        logger.error(f"Failed to get positions. Error: {error}")
                    return []

                # Convert positions tuple to list of dictionaries
                positions_list = []
                for position in positions:
                    positions_list.append({
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "open_price": position.price_open,
                        "current_price": position.price_current,
                        "sl": position.sl,
                        "tp": position.tp,
                        "profit": position.profit,
                        "swap": position.swap,
                        "type": position.type,  # 0=buy, 1=sell
                        "magic": position.magic,
                        "comment": position.comment,
                        "time": position.time,
                    })

                return positions_list

            except Exception as e:
                recovery_attempts += 1
                logger.error(f"Error getting positions: {str(e)}")

                if recovery_attempts <= max_recovery_attempts:
                    logger.warning(f"Attempting to reconnect (attempt {recovery_attempts}/{max_recovery_attempts})")
                    if self.initialize():
                        continue
                    time.sleep(1)
                else:
                    return []

        # If we get here, all recovery attempts failed
        return []

    def close_position(self, ticket: int) -> bool:
        """Close a specific position by ticket number."""
        if not self.connected:
            logger.error("MT5 not connected")
            return False

        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).ask if close_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).bid

        # Get the appropriate filling mode for this symbol
        filling_mode = self.get_symbol_filling_mode(pos.symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,  # Use appropriate filling mode for this symbol
        }

        result = mt5.order_send(request)
        if not result or (result.retcode not in (mt5.TRADE_RETCODE_DONE,) and
                          not (result.retcode == 1 and result.comment == "Success")):
            logger.error(f"Failed to close position {ticket}. Error: {result}")
            return False

        return True

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Get historical data from MT5."""
        if not self.connected:
            logger.error("MT5 not connected")
            return None

        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }

        tf = timeframe_map.get(timeframe)
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        try:
            # Ensure symbol is available
            if not self.is_symbol_available(symbol):
                logger.error(f"Symbol {symbol} is not available in MT5")
                return None

            # Get historical data
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.error(f"Failed to get historical data for {symbol} {timeframe}. Error: {error}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Add tick volume as volume
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']

            logger.debug(f"Retrieved {len(df)} historical bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} {timeframe}: {str(e)}")
            return None

    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.connected:
            try:
                # Check if we're in the middle of a critical operation
                stack = traceback.extract_stack()
                skip_shutdown = False

                # Operations during which we should not shutdown
                critical_operations = [
                    'change_signal_generator',
                    'get_market_data',
                    'get_rates',
                    'main_loop',
                    'get_account_info',
                    'get_open_positions',
                    'symbol_select',
                    'get_historical_data'
                ]

                for frame in stack:
                    if any(op in frame.name for op in critical_operations):
                        # Skip shutdown during critical operations
                        logger.debug(f"Skipping MT5 connection shutdown during critical operation: {frame.name}")
                        skip_shutdown = True
                        break

                if skip_shutdown:
                    return

                # Only perform shutdown for explicit manual shutdown requests
                # Check if this is called from manage_open_trades or process_signals
                for frame in stack:
                    if 'manage_open_trades' in frame.name or 'process_signals' in frame.name:
                        logger.debug("Skipping MT5 connection shutdown during trading operations")
                        return

                # Check if called from a method that should allow shutdown
                safe_shutdown_contexts = ['stop', 'shutdown', '__del__', 'recover_mt5_connection']
                allow_shutdown = any(context in frame.name for frame in stack for context in safe_shutdown_contexts)

                if not allow_shutdown:
                    logger.debug("Skipping automatic MT5 shutdown to maintain connection stability")
                    return

                logger.info("Performing explicit MT5 connection shutdown")
                if hasattr(mt5, 'shutdown'):
                    mt5.shutdown()
                self.connected = False
                logger.info("MT5 connection closed")
            except Exception as e:
                logger.error(f"Error during MT5 shutdown: {str(e)}")

    def modify_position(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify the stop loss and take profit of an open position using the MT5 API."""
        if not self.connected:
            logger.error("MT5 not connected")
            return False

        # Verify position exists
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position {ticket} not found")
            return False

        position = position[0]

        # Validate new levels
        symbol_info = mt5.symbol_info(position.symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {position.symbol}")
            return False

        min_stop_distance = symbol_info.point * symbol_info.trade_stops_level
        current_price = position.price_current

        # Validate stop loss
        if position.type == mt5.ORDER_TYPE_BUY:
            if new_sl >= current_price - min_stop_distance:
                logger.error(f"Invalid stop loss: too close to current price. Min distance: {min_stop_distance}")
                return False
        else:  # SELL
            if new_sl <= current_price + min_stop_distance:
                logger.error(f"Invalid stop loss: too close to current price. Min distance: {min_stop_distance}")
                return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": new_sl,
            "tp": new_tp,
            "deviation": 20,  # Increased deviation
            "magic": 234000,
            "comment": "Modify position SL/TP",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Try multiple times with increasing deviation
        max_retries = 3
        current_deviation = 20

        for attempt in range(max_retries):
            request["deviation"] = current_deviation
            result = mt5.order_send(request)

            if result is None:
                logger.error(f"Modification failed. Error: {mt5.last_error()}")
                current_deviation += 10
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Successfully modified position {ticket} SL/TP")
                return True

            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logger.warning(f"Requote detected on attempt {attempt + 1}")
                current_deviation += 10
                continue

            logger.error(f"Modification failed. Retcode: {result.retcode}, Comment: {result.comment}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                continue

        return False

    def get_spread(self, symbol):
        """Get current spread for a symbol in pips."""
        try:
            # Calculate pip multiplier dynamically using the symbol's point value
            sym_info = mt5.symbol_info(symbol)
            if not sym_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return float('inf')
            multiplier = 1 / sym_info.point

            # Get current symbol info using MT5 directly
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick data for {symbol}")
                return float('inf')

            spread = (tick.ask - tick.bid) * multiplier
            logger.debug(f"Current spread for {symbol}: {spread} pips")
            return spread
        except Exception as e:
            logger.error(f"Error getting spread: {str(e)}")
            return float('inf')

    def get_min_stop_distance(self, symbol: str, use_fallback: bool = False) -> float:
        """
        Get the minimum stop distance for a symbol in points. Includes a configurable buffer.
        """
        if use_fallback:
            return self._get_fallback_min_stop_distance(symbol)

        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"Could not get symbol info for {symbol}, using fallback for min stop distance.")
            return self._get_fallback_min_stop_distance(symbol)

        # trade_stops_level is in points, not price.
        min_stop_level_points = symbol_info.trade_stops_level
        
        # If the broker reports 0, it's unreliable. Use fallback.
        if min_stop_level_points == 0:
            logger.debug(f"Broker reported 0 for trade_stops_level for {symbol}. Using fallback.")
            return self._get_fallback_min_stop_distance(symbol)

        # Add the configured buffer to the stop level
        buffer_percent = RISK_MANAGER_CONFIG.get('stop_level_buffer_percent', 0.1) # Default to 10%
        buffered_stop_level = min_stop_level_points * (1 + buffer_percent)
        
        point_value = self.get_point_value(symbol)
        min_stop_distance_price = buffered_stop_level * point_value
        
        logger.trace(f"[{symbol}] Min stop distance: {min_stop_level_points} points, with {buffer_percent*100}% buffer -> {buffered_stop_level:.2f} points ({min_stop_distance_price:.5f} price units)")
        
        return min_stop_distance_price

    def _get_fallback_min_stop_distance(self, symbol: str) -> float:
        """
        Fallback method to calculate a reasonable minimum stop distance based on ATR or price percentage.
        This is used when broker-provided values are unreliable.

        Args:
            symbol: The trading symbol

        Returns:
            float: A fallback minimum stop distance
        """
        # Assign fallback values based on symbol type and typical price ranges

        # Precious metals
        if any(symbol.startswith(prefix) for prefix in ["GOLD", "XAU", "SILVER", "XAG", "PLATINUM", "XPT"]):
            if "GOLD" in symbol or "XAU" in symbol:
                return 0.5  # Gold typically needs larger stops (50 cents)
            elif "SILVER" in symbol or "XAG" in symbol:
                return 0.02  # Silver has lower price so smaller stop
            elif "PLATINUM" in symbol or "XPT" in symbol:
                return 0.5
            return 0.2  # Default for other metals

        # Stock indices
        elif any(symbol.startswith(prefix) for prefix in ["US30", "US500", "USTEC", "SPX", "NDX", "DJI", "DAX", "FTSE"]):
            if symbol.startswith("US30") or "DJI" in symbol:
                return 2.0  # Dow Jones-based products (larger value)
            elif "DAX" in symbol:
                return 1.0  # DAX Index
            elif "FTSE" in symbol:
                return 1.0  # FTSE Index
            return 0.5  # Default for other indices

        # Synthetic/Deriv indices
        elif any(symbol.startswith(prefix) for prefix in ["Crash", "Boom", "Jump", "Volatility", "Range", "Step"]):
            # Get first digits from symbol name
            digits_part = ''.join(filter(str.isdigit, symbol))
            if digits_part:
                index_value = int(digits_part[:3])  # Take first three digits
                if index_value >= 500:
                    return 0.10  # Higher index values need larger stops
                elif index_value >= 100:
                    return 0.05  # Medium index values
                else:
                    return 0.02  # Smaller index values
            # Default for synthetic indices if no digits found
            if "Crash" in symbol or "Boom" in symbol:
                return 0.05
            elif "Jump" in symbol or "Volatility" in symbol:
                return 0.02
            return 0.03  # Default for other synthetic indices

        # Cryptocurrency pairs
        elif any(crypto in symbol for crypto in ["BTC", "ETH", "LTC", "XRP", "DOGE", "BCH", "XMR"]):
            if "BTC" in symbol:
                return 50.0  # Bitcoin needs large stops
            elif "ETH" in symbol:
                return 5.0  # Ethereum
            return 1.0  # Default for other cryptos

        # Currency pairs
        elif symbol.endswith("JPY") or "JPY" in symbol:
            return 0.01  # JPY pairs typically use 2 decimal places
        elif any(symbol.startswith(prefix) for prefix in ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF"]):
            # Regular forex pairs with 4-5 decimal places
            return 0.0003

        # Default fallback if none of the above patterns match
        return 0.0003  # Conservative default for unknown symbols


    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information from MT5.

        Args:
            symbol: The trading symbol to get information for

        Returns:
            Symbol information object or None if not found/error
        """
        if not self.connected:
            logger.error("MT5 not connected")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
            return None

        return symbol_info

    def get_order_history(self, ticket=None, symbol=None, days=3):
        """Get historical orders from MT5

        Args:
            ticket (int, optional): Specific order ticket to retrieve
            symbol (str, optional): Symbol to get history for
            days (int, optional): Number of days to look back. Defaults to 3.

        Returns:
            list: List of historical orders
        """
        if not self.connected:
            if not self.initialize():
                logger.error("Failed to connect to MT5 when retrieving order history")
                return []

        try:
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()

            # Log detailed date range information
            logger.info(f"Fetching order history from {from_date.strftime('%Y-%m-%d %H:%M:%S')} to {to_date.strftime('%Y-%m-%d %H:%M:%S')} ({days} days)")

            # Define parameters
            history_orders = None

            # Different call options based on parameters
            if ticket is not None:
                logger.info(f"Filtering order history by ticket: {ticket}")
                # Call with ticket parameter
                history_orders = mt5.history_orders_get(ticket=ticket)
            elif symbol is not None:
                logger.info(f"Filtering order history by symbol: {symbol}")
                # Call with symbol filter as a group parameter
                history_orders = mt5.history_orders_get(from_date, to_date, group=symbol)
            else:
                # Default call with just date range
                # Use positional parameters for from_date and to_date
                logger.debug(f"MT5 history_orders_get request with date range: {from_date} to {to_date}")
                history_orders = mt5.history_orders_get(from_date, to_date)

            if history_orders is None:
                error = mt5.last_error()
                logger.warning(f"MT5 history_orders_get returned None. Error: {error}")
                # Return empty list instead of continuing to try to iterate
                return []
            elif len(history_orders) == 0:
                logger.warning(f"No order history found for the specified period ({days} days)")

                # Try to get deals history instead, which should have the profit information
                deals_history = None

                if ticket is not None:
                    deals_history = mt5.history_deals_get(ticket=ticket)
                elif symbol is not None:
                    deals_history = mt5.history_deals_get(from_date, to_date, group=symbol)
                else:
                    logger.debug(f"MT5 history_deals_get request with date range: {from_date} to {to_date}")
                    deals_history = mt5.history_deals_get(from_date, to_date)

                if deals_history is None:
                    error = mt5.last_error()
                    logger.warning(f"MT5 history_deals_get returned None. Error: {error}")
                    # Return empty list instead of trying further
                    return []
                elif len(deals_history) == 0:
                    logger.warning(f"No deals history found for the specified period ({days} days)")

                    # Try a larger date range as a test
                    extended_from_date = datetime.now() - timedelta(days=days*2)
                    logger.info(f"Testing with extended date range: {extended_from_date.strftime('%Y-%m-%d %H:%M:%S')} to {to_date.strftime('%Y-%m-%d %H:%M:%S')} ({days*2} days)")
                    extended_deals_history = mt5.history_deals_get(extended_from_date, to_date)

                    if extended_deals_history is not None and len(extended_deals_history) > 0:
                        logger.info(f"Found {len(extended_deals_history)} deals in extended date range ({days*2} days)")

                    # Check open positions to compare
                    open_positions = mt5.positions_get()
                    if open_positions is not None:
                        logger.info(f"Currently have {len(open_positions)} open positions")

                    return []

                # Convert deals to dict format, which will have profit
                result = []
                for deal in deals_history:
                    # Check if this deal matches our order
                    if ticket is not None and deal.order != ticket:
                        continue

                    result.append({
                        "ticket": deal.order,  # Use order number
                        "time": deal.time,
                        "time_close": deal.time,
                        "symbol": deal.symbol,
                        "type": deal.type,
                        "volume": deal.volume,
                        "price": deal.price,
                        "price_current": deal.price,
                        "sl": 0,  # Not available in deals
                        "tp": 0,  # Not available in deals
                        "state": 0,  # Not applicable for deals
                        "profit": deal.profit
                    })

                logger.info(f"Converted {len(result)} deals to trade history format")
                return result

            # Convert orders to list of dictionaries
            result = []
            for order in history_orders:
                # For each order, try to find the corresponding deal to get profit
                if hasattr(order, 'ticket'):
                    # Try to get deals for this order
                    deals = mt5.history_deals_get(order=order.ticket)
                    profit = 0
                    if deals and len(deals) > 0:
                        # Sum up profits from all deals for this order
                        profit = sum(deal.profit for deal in deals if hasattr(deal, 'profit'))

                    result.append({
                        "ticket": order.ticket,
                        "time": order.time_setup,
                        "time_close": getattr(order, 'time_done', order.time_setup),
                        "symbol": order.symbol,
                        "type": order.type,
                        "volume": order.volume_initial,
                        "price": order.price_open,
                        "price_current": getattr(order, 'price_current', order.price_open),
                        "sl": order.sl,
                        "tp": order.tp,
                        "state": order.state,
                        "profit": profit  # Use accumulated profit from deals
                    })

            logger.info(f"Converted {len(result)} orders to trade history format")
            return result

        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            logger.error(traceback.format_exc())
            if ticket:
                logger.error(f"Failed to get history for ticket {ticket}")
            return []

    def get_account_history(self, days=3):
        """Get historical account data (balance, equity, etc.)

        Args:
            days (int, optional): Number of days to look back. Defaults to 3.

        Returns:
            list: List of daily account balance records
        """
        if not self.connected:
            if not self.initialize():
                logger.error("Failed to connect to MT5 when retrieving account history")
                return []

        try:
            # Get deals for the specified period
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()

            # Add logging for date range
            logger.info(f"Fetching account history from {from_date.strftime('%Y-%m-%d %H:%M:%S')} to {to_date.strftime('%Y-%m-%d %H:%M:%S')} ({days} days)")

            # Use positional parameters for date range
            logger.debug(f"MT5 history_deals_get request with date range: {from_date} to {to_date}")
            deals = mt5.history_deals_get(from_date, to_date)

            if deals is None or len(deals) == 0:
                logger.warning(f"No deals found for the past {days} days")

                # Try a longer period as a test
                test_from_date = datetime.now() - timedelta(days=days*2)
                logger.info(f"Testing with longer period: {test_from_date.strftime('%Y-%m-%d %H:%M:%S')} to {to_date.strftime('%Y-%m-%d %H:%M:%S')}")
                test_deals = mt5.history_deals_get(test_from_date, to_date)
                if test_deals is not None and len(test_deals) > 0:
                    logger.info(f"Found {len(test_deals)} deals when looking back {days*2} days")

                return self._generate_balance_history(days)

            # Log number of deals found
            logger.info(f"Found {len(deals)} deals for the past {days} days")

            # Group deals by day
            daily_balance = {}
            account_info = self.get_account_info()
            current_balance = account_info.get("balance", 0) if account_info is not None else 0

            # Start with current balance and work backwards
            for deal in sorted(deals, key=lambda x: x.time, reverse=True):
                # Convert timestamp to datetime if it's an integer
                if isinstance(deal.time, int):
                    deal_datetime = datetime.fromtimestamp(deal.time)
                else:
                    deal_datetime = deal.time

                deal_date = deal_datetime.date()
                deal_date_str = deal_date.strftime("%Y-%m-%d")

                # Subtract the profit to get the balance before this deal
                if deal.profit != 0:
                    current_balance -= deal.profit

                if deal_date_str not in daily_balance:
                    daily_balance[deal_date_str] = {
                        "date": deal_date_str,
                        "balance": current_balance,
                        "profit_loss": 0,
                        "drawdown": 0,
                        "win_rate": 0
                    }

            # Ensure we have entries for all days, even without trades
            result = self._fill_missing_days(daily_balance, days)

            # Calculate daily profit/loss, drawdown and win rate
            self._calculate_metrics(result)

            return sorted(result, key=lambda x: x["date"])

        except Exception as e:
            logger.error(f"Error getting account history: {e}")
            return self._generate_balance_history(days)

    def _generate_balance_history(self, days=3):
        """Generate empty balance history when no data is available

        Args:
            days (int): Number of days

        Returns:
            list: Generated balance history with current balance
        """
        result = []
        account_info = self.get_account_info()
        current_balance = account_info.get("balance", 0) if account_info is not None else 0

        # Create entries for each day
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-i-1)).date()
            date_str = date.strftime("%Y-%m-%d")

            result.append({
                "date": date_str,
                "balance": current_balance,
                "profit_loss": 0,
                "drawdown": 0,
                "win_rate": 0
            })

        return result

    def _fill_missing_days(self, daily_balance, days):
        """Fill in missing days in the balance history

        Args:
            daily_balance (dict): Existing balance data by date
            days (int): Number of days to include

        Returns:
            list: Complete balance history
        """
        result = []

        # Get the last balance value (most recent)
        last_balance = next(iter(daily_balance.values()))["balance"] if daily_balance else self.get_account_info().get("balance", 0)

        # Create entries for each day
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-i-1)).date()
            date_str = date.strftime("%Y-%m-%d")

            if date_str in daily_balance:
                result.append(daily_balance[date_str])
            else:
                result.append({
                    "date": date_str,
                    "balance": last_balance,
                    "profit_loss": 0,
                    "drawdown": 0,
                    "win_rate": 0
                })

        return result

    def _calculate_metrics(self, balance_history):
        """Calculate metrics for balance history

        Args:
            balance_history (list): Balance history to update

        Returns:
            None: Updates the balance_history in place
        """
        if not balance_history:
            return

        # Get the first balance as baseline
        baseline_balance = balance_history[0]["balance"]
        peak_balance = baseline_balance

        # Calculate daily profit/loss and drawdown
        for i, day in enumerate(balance_history):
            if i > 0:
                prev_balance = balance_history[i-1]["balance"]
                day["profit_loss"] = day["balance"] - prev_balance

                # Update peak balance
                if day["balance"] > peak_balance:
                    peak_balance = day["balance"]

                # Calculate drawdown from peak
                if peak_balance > 0:
                    drawdown_pct = ((peak_balance - day["balance"]) / peak_balance) * 100
                    day["drawdown"] = drawdown_pct
                else:
                    day["drawdown"] = 0

                # For win rate, we need trade data - this is a placeholder
                # In a real system, you'd calculate this from actual trade results
                day["win_rate"] = 50 + (i * 2) # Placeholder that increases daily

    def get_error_info(self, error_code: Optional[int] = None) -> str:
        """
        Get detailed error information from MT5.

        Args:
            error_code: Optional specific error code to get description for.
                        If None, returns the last error from MT5

        Returns:
            str: Formatted error description
        """
        try:
            error_descriptions = {
                10004: "Trade server is busy",
                10006: "Request rejected",
                10007: "Request canceled by trader",
                10008: "Order already placed",
                10009: "Order placed",
                10010: "Request placed",
                10011: "Request executed",
                10012: "Request executed partially",
                10013: "Only part of the request was completed",
                10014: "Request processing error",
                10015: "Request canceled by timeout",
                10016: "Request canceled by dealer",
                10017: "Dealer processed the request",
                10018: "Request received and accepted for processing",
                10019: "Request executed partially",
                10020: "Request received and being processed",
                10021: "Request canceled due to connection problem",
                10022: "Request canceled due to re-quote",
                10023: "Request canceled due to order expiration",
                10024: "Request canceled due to fill conditions",
                10025: "Request not accepted for further processing",
                10026: "Request accepted for further processing",
                10027: "Request canceled due to order stop activation",
                10028: "Request cancelled due to position closure",
                10029: "Trade operation requested is not supported",
                10030: "Open volume exceeds limit",
                10031: "Server closed the connection",
                10032: "Server reopened the connection",
                10033: "Initial status",
                10034: "Partial close performed",
                10035: "No quotes to process request",
                10036: "Request to close the position rejected because of the hedge limitation",
                10038: "Modification request rejected, as a request to close this order is already in process",
                10039: "Execution request rejected, as a request to close this order is already in process",
                10040: "Close request rejected because the position is still not fully opened",
                10041: "Request rejected because the order is being processed for closure",
                10042: "Request rejected, as the order is not in a suitable state",
                10043: "Trade timeout",
                10044: "Trades quota has been exhausted",
                10045: "Execution is rejected due to the restrictions on the number of pending orders",
                10046: "Close volume exceeds the limit",
                10047: "Execution for the total volume has been denied",
                10048: "The execution of the request is possible only when the market is open",
                10049: "Limit order requote",
                10050: "Order volume is too big",
                10051: "Client terminal is not connected to the server",
                10052: "Operation is only available for live accounts",
                10053: "Reached order limits set by broker",
                10054: "Reached position limits set by broker",
                # Common MT5 error codes
                1: "Success, no error returned",
                2: "Common error",
                3: "Invalid trade parameters",
                4: "Trade server is busy",
                5: "Old version of the client terminal",
                6: "No connection with trade server",
                7: "Not enough rights",
                8: "Too frequent requests",
                9: "Malfunctional trade operation",
                64: "Account disabled",
                65: "Invalid account",
                128: "Trade timeout",
                129: "Invalid price",
                130: "Invalid stops",
                131: "Invalid trade volume",
                132: "Market is closed",
                133: "Trade is disabled",
                134: "Not enough money",
                135: "Price changed",
                136: "Off quotes",
                137: "Broker is busy",
                138: "Requote",
                139: "Order is locked",
                140: "Long positions only allowed",
                141: "Too many requests",
                145: "Modification denied because order is too close to market",
                146: "Trade context is busy",
                147: "Expirations are denied by broker",
                148: "Amount of open and pending orders has reached the limit",
                149: "Hedging is prohibited",
                150: "Prohibited by FIFO rules"
            }

            # If a specific error code was provided, return its description
            if error_code is not None:
                if error_code in error_descriptions:
                    return f"{error_code}: {error_descriptions[error_code]}"
                return f"Unknown error code: {error_code}"

            # Otherwise get the last error from MT5
            mt5_error = mt5.last_error()

            # Format depends on whether it's a tuple or a single value
            if isinstance(mt5_error, tuple) and len(mt5_error) >= 2:
                error_code, error_description = mt5_error[0], mt5_error[1]

                # Only create error message for actual errors (code != 0)
                if error_code != 0:
                    # Add detailed description if available
                    if isinstance(error_code, int) and error_code in error_descriptions:
                        return f"MT5 Error: {error_code} - {error_description} ({error_descriptions[error_code]})"
                    return f"MT5 Error: {error_code} - {error_description}"
                else:
                    # This is a "success" message, not an error
                    return "No error"
            elif isinstance(mt5_error, int):
                # Handle case where only error code is returned
                if mt5_error != 0:
                    if mt5_error in error_descriptions:
                        return f"MT5 Error Code: {mt5_error} ({error_descriptions[mt5_error]})"
                    return f"MT5 Error Code: {mt5_error}"
                else:
                    return "No error"
            else:
                # For any other format, return as string
                if mt5_error:
                    return f"MT5 Error: {mt5_error}"
                return "No error"
        except Exception as e:
            logger.error(f"Error getting MT5 error info: {str(e)}")
            return f"Error retrieving MT5 error: {str(e)}"

    def calculate_position_size(self, symbol, price=None, risk_amount=None, risk_percent=None, entry_price=None, stop_loss_price=None):
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Symbol to trade
            price: Current price (optional)
            risk_amount: Fixed amount to risk (optional)
            risk_percent: Percentage of balance to risk (optional)
            entry_price: Entry price for the trade (optional)
            stop_loss_price: Stop loss price (optional)

        Returns:
            float: Position size in lots
        """
        try:
            # Check connection
            if not self.connected:
                logger.error("MT5 not connected")
                return self.get_symbol_min_lot_size(symbol)  # Use new method for minimum lot size

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found in MT5")
                return self.get_symbol_min_lot_size(symbol)  # Use new method for minimum lot size

            # Ensure the symbol is selected
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return self.get_symbol_min_lot_size(symbol)  # Use new method for minimum lot size

            # Get the pip value
            digits = symbol_info.digits
            point = symbol_info.point
            contract_size = symbol_info.trade_contract_size

            # For Forex pairs, a pip is usually the 4th decimal place for 5-digit quotes
            # For JPY pairs, a pip is usually the 2nd decimal place for 3-digit quotes
            pip_multiplier = 10 if digits == 3 or digits == 5 else 1
            pip_value = point * pip_multiplier

            # Get entry price if not provided
            if entry_price is None:
                # Use the latest tick if price not specified
                if price is None:
                    tick = self.get_last_tick(symbol)
                    if tick is None:
                        logger.error(f"Failed to get last tick for {symbol}")
                        return 0.01
                    price = tick.get('ask', 0)
                entry_price = price

            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.01

            # Determine risk amount
            if risk_amount is None and risk_percent is not None:
                balance = account_info.balance
                risk_amount = balance * risk_percent / 100.0
            elif risk_amount is None:
                logger.warning("Neither risk_amount nor risk_percent provided, using minimal position size")
                return 0.01

            # Calculate stop loss distance
            if stop_loss_price is not None:
                stop_loss_pips = abs(entry_price - stop_loss_price) / pip_value
            else:
                logger.warning("No stop loss price provided, using default 10 pips")
                stop_loss_pips = 10.0

            # If stop loss distance is zero or very small, use default
            if stop_loss_pips < 0.1:
                logger.warning(f"Very small stop loss distance: {stop_loss_pips} pips, using default 10 pips")
                stop_loss_pips = 10.0

            # Get conversion rate to account currency if needed
            account_currency = account_info.currency
            conversion_rate = 1.0

            # Calculate position size based on risk
            position_size = risk_amount / (stop_loss_pips * pip_value * contract_size * conversion_rate)

            # Adjust to symbol's volume_step
            volume_step = symbol_info.volume_step
            position_size = math.floor(position_size / volume_step) * volume_step

            # Ensure minimum size
            position_size = max(position_size, symbol_info.volume_min)

            # Ensure maximum size
            position_size = min(position_size, symbol_info.volume_max)

            logger.info(f"Calculated position size for {symbol}: {position_size} lots (risk: {risk_amount})")

            # Get minimum lot size and volume step for the symbol
            min_lot_size = self.get_symbol_min_lot_size(symbol)

            # Normalize the lot size according to the symbol's requirements
            position_size = self.normalize_volume(symbol, position_size)

            # Ensure position size is not less than minimum
            if position_size < min_lot_size:
                logger.info(f"Calculated position size {position_size} is less than minimum {min_lot_size} for {symbol}, using minimum")
                position_size = min_lot_size

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Return minimum position size on error

    def get_symbol_min_lot_size(self, symbol: str) -> float:
        """
        Get the minimum permitted lot size for a specified symbol.

        Args:
            symbol (str): The trading symbol to check

        Returns:
            float: Minimum lot size for the symbol, or 0.01 as fallback
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected")
                return 0.01

            # Get symbol info from MT5
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.01

            # Extract minimum volume and volume step
            min_volume = symbol_info.volume_min
            volume_step = symbol_info.volume_step

            logger.debug(f"Symbol {symbol} min volume: {min_volume}, volume step: {volume_step}")

            # Ensure the value is valid
            if min_volume <= 0:
                logger.warning(f"Symbol {symbol} has invalid min_volume {min_volume}, using 0.01")
                return 0.01

            return min_volume

        except Exception as e:
            logger.error(f"Error getting min lot size for {symbol}: {str(e)}")
            return 0.01

    def normalize_volume(self, symbol: str, volume: float) -> float:
        """
        Normalize the volume to be within the allowed range and aligned with volume_step.

        Args:
            symbol (str): The trading symbol
            volume (float): The desired volume

        Returns:
            float: The normalized volume that is valid for the symbol
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected")
                return 0.01

            # Get symbol info from MT5
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.01

            # Extract volume constraints
            min_volume = symbol_info.volume_min
            max_volume = symbol_info.volume_max
            volume_step = symbol_info.volume_step

            logger.debug(f"Symbol {symbol} volume constraints: min={min_volume}, max={max_volume}, step={volume_step}")

            # Check if volume is below minimum
            if volume < min_volume:
                logger.warning(f"Volume {volume} is below minimum {min_volume} for {symbol}, adjusting to minimum")
                return min_volume

            # Check if volume is above maximum
            if volume > max_volume:
                logger.warning(f"Volume {volume} exceeds maximum {max_volume} for {symbol}, adjusting to maximum")
                return max_volume

            # Normalize volume to the nearest volume_step
            normalized_volume = round(volume / volume_step) * volume_step

            # Ensure it's still within bounds after rounding
            if normalized_volume < min_volume:
                normalized_volume = min_volume
            elif normalized_volume > max_volume:
                normalized_volume = max_volume

            # Ensure precision: round to acceptable number of decimal places
            # Most brokers handle up to 2 decimal places for lot sizes
            normalized_volume = round(normalized_volume, 2)

            return normalized_volume

        except Exception as e:
            logger.error(f"Error normalizing volume for {symbol}: {str(e)}")
            return 0.01

    # NEW METHOD: Periodic Data Fetching
    async def fetch_data_periodically(self, symbols, timeframes, callback, interval_seconds=60):
        """
        Periodically fetch data for multiple symbols and timeframes.

        Args:
            symbols (list): List of symbols to fetch data for
            timeframes (list): List of timeframes to fetch data for
            callback (callable): Function to call with fetched data
            interval_seconds (int): Base interval in seconds between fetches

        This method will run indefinitely until canceled and will fetch data
        for each symbol and timeframe combination at appropriate intervals.
        """
        if not symbols or not timeframes:
            logger.error("No symbols or timeframes provided for periodic data fetching")
            return

        logger.info(f"Starting periodic data fetching for {len(symbols)} symbols and {len(timeframes)} timeframes")

        # Create a mapping of timeframes to their intervals in seconds
        tf_intervals = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H4': 14400,
            'D1': 86400,
        }

        # Keep track of when we last fetched data for each symbol/timeframe
        last_fetch = {}
        for symbol in symbols:
            for timeframe in timeframes:
                last_fetch[(symbol, timeframe)] = 0  # Initialize to 0 to fetch immediately

        # Create a stop event to allow canceling this task
        self._fetch_data_stop_event = asyncio.Event()

        try:
            while not self._fetch_data_stop_event.is_set():
                current_time = time.time()

                # Check for each symbol and timeframe if it's time to fetch
                for symbol in symbols:
                    for timeframe in timeframes:
                        tf_interval = tf_intervals.get(timeframe, interval_seconds)

                        # Fetch if enough time has passed since last fetch
                        if current_time - last_fetch.get((symbol, timeframe), 0) >= tf_interval:
                            try:
                                # Fetch the data using get_market_data instead of get_historical_data
                                # Default to 1000 candles or use a reasonable lookback for the timeframe
                                num_candles = 1000
                                if timeframe.startswith('M'):
                                    # For minute timeframes, use more candles for shorter timeframes
                                    try:
                                        minutes = int(timeframe[1:])
                                        num_candles = min(2000, max(1000, int(5000 / minutes)))
                                    except ValueError:
                                        num_candles = 1000
                                elif timeframe.startswith('H'):
                                    # For hourly timeframes, fewer candles are typically needed
                                    num_candles = 500

                                # Call get_market_data which doesn't require date parameters
                                data = self.get_market_data(symbol, timeframe, num_candles)

                                # Update last fetch time
                                last_fetch[(symbol, timeframe)] = current_time

                                # Call the callback with the fetched data
                                if data is not None and callback:
                                    # Handle both sync and async callbacks
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(symbol, timeframe, data)
                                    else:
                                        callback(symbol, timeframe, data)
                            except Exception as e:
                                logger.error(f"Error fetching data for {symbol}/{timeframe}: {str(e)}")

                # Sleep for a short time before checking again
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Periodic data fetching task was cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in periodic data fetching: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self._fetch_data_stop_event = None
            logger.info("Periodic data fetching stopped")

    def _get_mt5_timeframe(self, timeframe: str) -> int:
        """
        Convert string timeframe to MT5 timeframe constant.

        Args:
            timeframe: String timeframe (e.g., "M15", "H1")

        Returns:
            MT5 timeframe constant
        """
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }

        tf = timeframe_map.get(timeframe)
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            # Default to H1 if invalid
            return mt5.TIMEFRAME_H1

        return tf

    def get_position_by_ticket(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Get position details by ticket number.

        Args:
            ticket: The position ticket number

        Returns:
            Optional[Dict[str, Any]]: Position details or None if not found
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected")
                return None

            # Get position from MT5
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return None

            # Convert position object to dictionary
            pos = position[0]
            return {
                "ticket": getattr(pos, "ticket", 0),
                "symbol": getattr(pos, "symbol", ""),
                "type": getattr(pos, "type", 0),
                "volume": getattr(pos, "volume", 0.0),
                "price_open": getattr(pos, "price_open", 0.0),
                "price_current": getattr(pos, "price_current", 0.0),
                "sl": getattr(pos, "sl", 0.0),
                "tp": getattr(pos, "tp", 0.0),
                "profit": getattr(pos, "profit", 0.0),
                "comment": getattr(pos, "comment", ""),
                "time": getattr(pos, "time", 0),
                "magic": getattr(pos, "magic", 0),
                "swap": getattr(pos, "swap", 0.0),
                "commission": getattr(pos, "commission", 0.0)
            }
        except Exception as e:
            logger.error(f"Error getting position by ticket {ticket}: {str(e)}")
            return None

    def is_connected(self) -> bool:
        """Check if MT5 is connected."""
        try:
            if not hasattr(mt5, 'terminal_info'):
                return False

            # Get terminal info to verify connection
            terminal_info = mt5.terminal_info()

            if terminal_info is None:
                self.connected = False
                return False

            # Check connected flag from terminal info
            is_connected = getattr(terminal_info, 'connected', False)

            # Log terminal information for diagnostics
            if is_connected:
                logger.debug(f"MT5 connected with terminal info: build={getattr(terminal_info, 'build', 'N/A')}, company={getattr(terminal_info, 'company', 'N/A')}")
                # Check if history is enabled in the terminal
                max_bars = getattr(terminal_info, 'maxbars', 0)
                community_account = getattr(terminal_info, 'community_account', False)
                community_connection = getattr(terminal_info, 'community_connection', False)

                logger.debug(f"MT5 terminal settings: max_bars={max_bars}, community_account={community_account}, community_connection={community_connection}")

                if max_bars < 1000:
                    logger.warning(f"MT5 terminal max_bars setting is low: {max_bars}. This may limit history data availability.")

                # Try to get available symbol count as an additional connection test
                try:
                    symbols = mt5.symbols_get()
                    if symbols is not None:
                        logger.debug(f"MT5 reports {len(symbols)} available symbols")
                except Exception as e:
                    logger.warning(f"Could not get symbols list: {str(e)}")

            # Update our internal state
            self.connected = bool(is_connected)

            return bool(is_connected)
        except Exception as e:
            logger.error(f"Error checking MT5 connection: {str(e)}")
            logger.error(traceback.format_exc())
            self.connected = False
            return False

    def get_free_margin(self) -> float:
        """Get free margin from account info."""
        account_info = self.get_account_info()
        if not account_info:
            return 0.0

        # Use get() method instead of direct attribute access
        return float(account_info.get("margin_free", 0.0))

    async def stop(self):
        """
        Stop all running tasks and disconnect from MT5
        """
        # Stop periodic data fetching if active
        if hasattr(self, '_fetch_data_stop_event') and self._fetch_data_stop_event:
            self._fetch_data_stop_event.set()

        # Shutdown MT5 connection
        if self.connected:
            try:
                mt5.shutdown()
                self.connected = False
                logger.info("MT5 connection closed")
            except Exception as e:
                logger.error(f"Error during MT5 shutdown: {str(e)}")

    def get_last_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest tick data for a symbol.

        Args:
            symbol: Symbol to get tick data for

        Returns:
            Dictionary with tick data or None if error
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected")
                return None

            # Select the symbol to make sure it's available
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None

            # Get the latest tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get tick data for {symbol}: {mt5.last_error()}")
                return None

            # Convert to dictionary
            tick_dict = {
                "symbol": symbol,
                "time": tick.time,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time_msc": getattr(tick, "time_msc", 0),
                "flags": getattr(tick, "flags", 0),
                "volume_real": getattr(tick, "volume_real", 0.0)
            }

            return tick_dict

        except Exception as e:
            logger.error(f"Error getting last tick for {symbol}: {str(e)}")
            return None

    def get_symbol_filling_mode(self, symbol: str) -> int:
        """
        Determine the appropriate filling mode for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            int: The appropriate filling mode (MT5 constant)
        """
        try:
            # Get supported filling modes
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return mt5.ORDER_FILLING_FOK  # Default to FOK

            # Check the symbol's supported filling modes
            filling_mode = symbol_info.filling_mode
            logger.debug(f"Symbol {symbol} supports filling modes: {filling_mode}")

            # Determine the appropriate filling mode based on what's supported
            if (filling_mode & mt5.ORDER_FILLING_IOC) == mt5.ORDER_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            elif (filling_mode & mt5.ORDER_FILLING_FOK) == mt5.ORDER_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            elif (filling_mode & mt5.ORDER_FILLING_RETURN) == mt5.ORDER_FILLING_RETURN:
                return mt5.ORDER_FILLING_RETURN
            else:
                # Default to FOK if no specific filling mode is supported
                return mt5.ORDER_FILLING_FOK

        except Exception as e:
            logger.error(f"Error determining filling mode for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return mt5.ORDER_FILLING_FOK  # Default to FOK

    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if a symbol is available in MT5 and attempt to add it if not visible.

        Args:
            symbol: The trading symbol to check

        Returns:
            bool: True if the symbol is available, False otherwise
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected")
                return False

            # First check if the symbol exists in MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found in MT5")
                return False

            # If symbol exists but is not visible in MarketWatch, try to add it
            if not symbol_info.visible:
                logger.info(f"Symbol {symbol} not visible in MarketWatch, attempting to add it")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to add {symbol} to MarketWatch")
                    return False

                # Double-check it's now visible
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info or not symbol_info.visible:
                    logger.error(f"Symbol {symbol} still not visible after adding")
                    return False

            logger.debug(f"Symbol {symbol} is available and visible")
            return True

        except Exception as e:
            logger.error(f"Error checking symbol availability for {symbol}: {str(e)}")
            return False

    def get_point_value(self, symbol: str) -> float:
        """
        Get the point value for a given symbol.

        Args:
            symbol: The trading symbol.

        Returns:
            The point value (e.g., 0.00001 for EURUSD) or a default (0.0001) if not found.
        """
        if not self.connected:
            logger.error(f"MT5 not connected, cannot get point value for {symbol}")
            return 0.0001 # Default fallback

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Could not get symbol_info for {symbol} to determine point value. Using default.")
            return 0.0001

        try:
            point = getattr(symbol_info, 'point', None)
            if point is None or point == 0:
                logger.warning(f"Point value for {symbol} is None or zero in symbol_info. Symbol info details: Digits={getattr(symbol_info, 'digits', 'N/A')}, Spread={getattr(symbol_info, 'spread', 'N/A')}. Using default point value 0.0001.")
                # Attempt to calculate based on digits if point is missing/zero
                digits = getattr(symbol_info, 'digits', None)
                if digits is not None and digits > 0:
                    calculated_point = 1 / (10**digits)
                    logger.info(f"Calculated point value for {symbol} based on digits ({digits}): {calculated_point}")
                    return float(calculated_point)
                return 0.0001 # Default if digits also not helpful
            
            logger.debug(f"Point value for {symbol}: {point}")
            return float(point)
        except Exception as e:
            logger.error(f"Error accessing point value for {symbol} from symbol_info: {e}. Symbol info: {symbol_info}. Using default 0.0001.")
            return 0.0001

    def close_partial_position(self, ticket: int, volume_to_close: float) -> Optional[Dict[str, Any]]:
        """
        Closes a part of the position specified by the ticket.

        Args:
            ticket (int): The ticket of the position to partially close.
            volume_to_close (float): The volume (in lots) to close.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the trade result if successful, otherwise None.
        """
        if not self.connected:
            logger.error("MT5 not connected, cannot perform partial close.")
            return None

        position_info = self.get_position_by_ticket(ticket)
        if not position_info:
            logger.error(f"Cannot partially close position {ticket}: Not found.")
            return None

        symbol = position_info["symbol"]
        current_volume = position_info["volume"]

        if volume_to_close >= current_volume:
            logger.warning(f"Volume to close ({volume_to_close}) is equal to or greater than current volume ({current_volume}). Performing a full close instead.")
            close_result = self.close_position(ticket)
            return {"status": "full_close", "success": close_result} if close_result else None

        # Determine the correct order type for closing
        # If it's a BUY position, we need to SELL to close it.
        close_action = mt5.ORDER_TYPE_SELL if position_info["type"] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Get the current price for closing
        tick_info = self.get_last_tick(symbol)
        if not tick_info:
            logger.error(f"Could not get current price for {symbol} to perform partial close.")
            return None
        price = tick_info['bid'] if close_action == mt5.ORDER_TYPE_SELL else tick_info['ask']

        # Normalize the volume to close
        normalized_volume = self.normalize_volume(symbol, volume_to_close)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": normalized_volume,
            "type": close_action,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Partial close for ticket {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_symbol_filling_mode(symbol),
        }

        logger.info(f"Submitting partial close request for ticket {ticket}: Closing {normalized_volume} lots of {current_volume}.")
        
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Partial close for ticket {ticket} failed. Error: {self.get_error_info()}")
            return None
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Successfully executed partial close for ticket {ticket}. New volume should be {current_volume - normalized_volume}.")
            return result._asdict()
        else:
            logger.error(f"Partial close for ticket {ticket} failed with retcode {result.retcode}: {result.comment}")
            return None

    def get_market_book(self, symbol: str) -> Optional[List[Dict]]:
        """
        Subscribes to a symbol's market depth if needed and fetches the current order book.

        Args:
            symbol (str): The financial instrument to get the order book for.

        Returns:
            A list of order book entry dictionaries, or None if an error occurs.
        """
        if not self.connected:
            logger.error("MT5 not connected, cannot get market book.")
            return None

        # Subscribe if we haven't already
        if symbol not in self._active_subscriptions:
            if not mt5.market_book_add(symbol):
                error_code, error_message = mt5.last_error()
                # If subscription fails with a non-error code, it implies no market depth data is available.
                if error_code in [0, 1]:
                    logger.warning(
                        f"Could not subscribe to market depth for '{symbol}'. "
                        "This symbol likely does not provide Level 2 data."
                    )
                    logger.trace(f"Traceback for '{symbol}' market depth failure:\n{''.join(traceback.format_stack())}")
                else:
                    logger.error(f"Failed to subscribe to market depth for {symbol}: ({error_code}, '{error_message}')")
                return None

            logger.info(f"Subscribed to market depth for {symbol}.")
            self._active_subscriptions.add(symbol)
            
            # --- ADDED: A small delay to allow the book to populate on first subscription ---
            logger.trace(f"Pausing briefly for {symbol} market book to populate...")
            time.sleep(0.1) # 100ms warm-up delay

        # Get the book and convert to a list of dicts for easier use
        book = mt5.market_book_get(symbol)
        if book:
            return [item._asdict() for item in book]
        
        # If the subscription exists but the book is empty, return an empty list, not None.
        # This signals a valid subscription with no current data, not a failure.
        logger.trace(f"Subscription for {symbol} is active, but the market book is currently empty.")
        return []