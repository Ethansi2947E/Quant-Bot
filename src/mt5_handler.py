# mypy: ignore-errors
# pyright: reportAttributeAccessIssue=false
# flake8: noqa

import MetaTrader5 as mt5  # type: ignore
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any, cast
import time
import traceback
import json
import math
import sys
import re

from config.config import MT5_CONFIG

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

# Singleton instance for global reference
_mt5_handler_instance = None

class MT5Handler:
    def __init__(self):
        global _mt5_handler_instance
        # If an instance already exists, use it
        if _mt5_handler_instance is not None:
            logger.info("Using existing MT5Handler instance instead of creating a new one")
            # Copy attributes from existing instance
            self.__dict__ = _mt5_handler_instance.__dict__
            return
        
        _mt5_handler_instance = self
        self.connected = False
        self.initialize()
        self._last_error = None  # Add error tracking
    
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
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        num_candles: int = 1000
    ) -> Optional[pd.DataFrame]:
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
    
    def place_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: float,
        take_profit: float,
        comment: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Place a market order with proper validations."""
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
        
        # Map order type to MT5 constant
        if order_type == "BUY":
            action = mt5.ORDER_TYPE_BUY
            price = symbol_info.ask
        elif order_type == "SELL":
            action = mt5.ORDER_TYPE_SELL
            price = symbol_info.bid
        else:
            logger.error(f"Invalid order type: {order_type}")
            return None
            
        # Log current prices
        logger.debug(f"Current prices for {symbol}: Ask={symbol_info.ask}, Bid={symbol_info.bid}, Spread={symbol_info.ask - symbol_info.bid}")
        
        # Validate stop loss and take profit
        min_stop_distance = symbol_info.point * symbol_info.trade_stops_level
        
        if action == mt5.ORDER_TYPE_BUY:
            if stop_loss >= price - min_stop_distance:
                logger.error(f"Invalid stop loss for BUY order: SL ({stop_loss}) too close to entry ({price}). Min distance: {min_stop_distance}")
                return None
            if take_profit <= price + min_stop_distance:
                logger.error(f"Invalid take profit for BUY order: TP ({take_profit}) too close to entry ({price}). Min distance: {min_stop_distance}")
                return None
        else:  # SELL
            if stop_loss <= price + min_stop_distance:
                logger.error(f"Invalid stop loss for SELL order: SL ({stop_loss}) too close to entry ({price}). Min distance: {min_stop_distance}")
                return None
            if take_profit >= price - min_stop_distance:
                logger.error(f"Invalid take profit for SELL order: TP ({take_profit}) too close to entry ({price}). Min distance: {min_stop_distance}")
                return None
        
        # Log original volume request
        logger.info(f"Requested position size: {volume:.4f} lots")
        
        # Adjust position size based on available margin
        adjusted_volume = self.adjust_position_size(symbol, volume, price)
        
        # Enhanced margin error handling
        if adjusted_volume == 0:
            logger.error(f"Cannot place order: No margin available for {symbol}")
            
            # Calculate how much margin would be needed
            try:
                contract_size = symbol_info.trade_contract_size
                leverage = account_info.leverage if account_info else 100
                estimated_margin = (price * contract_size * volume) / leverage
                
                logger.error(f"Estimated margin needed: {estimated_margin:.2f}, but free margin is only: {account_info.margin_free:.2f}" if account_info else "Account info unavailable")
                logger.error(f"Consider reducing position size or depositing more funds")
            except Exception as e:
                logger.error(f"Error while calculating estimated margin: {str(e)}")
                
            return None
        
        # If volume was adjusted, log the reason
        if adjusted_volume < volume:
            logger.warning(f"Position size adjusted from {volume:.4f} to {adjusted_volume:.4f} lots due to margin constraints")
        
        # Round volume to valid step size
        adjusted_volume = round(adjusted_volume / symbol_info.volume_step) * symbol_info.volume_step
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": adjusted_volume,
            "type": action,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Try multiple times with increasing deviation
        max_retries = 3
        for attempt in range(max_retries):
            result = mt5.order_send(request)  # type: ignore
            if result is None:
                logger.error(f"Failed to send order: {mt5.last_error()}")  # type: ignore
                continue
                
            logger.debug(f"Order result: {json.dumps(result._asdict(), default=str)}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 10009:
                logger.info(f"Order executed successfully: Ticket {result.order}")
                return {
                    "ticket": result.order,
                    "volume": result.volume,
                    "price": result.price,
                    "comment": comment
                }
            
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logger.warning(f"Requote detected on attempt {attempt + 1}")
                # Update price and increase deviation
                tick = mt5.symbol_info_tick(symbol)  # type: ignore
                if tick:
                    request["price"] = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
                request["deviation"] += 10
                continue
                
            logger.error(f"Order failed with error code {result.retcode}")
            return None
            
        logger.error("Failed to place order after all retries")
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
            "type_filling": mt5.ORDER_FILLING_IOC,
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
        
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        if rates is None:
            logger.error(f"Failed to get historical data. Error: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Add tick volume as volume
        if 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
        
        return df

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
    
    def __del__(self):
        """Cleanup MT5 connection - but only when explicitly requested or at final program termination."""
        try:
            # Skip all destructor shutdowns except during complete program termination
            # This helps prevent connection issues during normal operations
            if not sys.is_finalizing():
                logger.debug("Skipping MT5 shutdown in __del__ during normal operation")
                return
                
            # Only shutdown if actually connected
            if hasattr(self, 'connected') and self.connected:
                try:
                    if hasattr(mt5, 'shutdown'):
                        mt5.shutdown()
                    logger.info("MT5 connection closed during program termination")
                except Exception as e:
                    if not sys.is_finalizing():
                        logger.error(f"Error during MT5 shutdown in __del__: {str(e)}")
        except Exception as e:
            # Don't log during interpreter shutdown
            if not sys.is_finalizing():
                logger.error(f"Error in MT5Handler.__del__: {str(e)}")

    async def get_rates(self, symbol: str, timeframe: str, num_candles: int = 1000) -> Optional[pd.DataFrame]:
        """Async wrapper around get_market_data for compatibility."""
        try:
            logger.debug(f"Fetching {num_candles} candles of {timeframe} data for {symbol}")
            
            # Check symbol availability before fetching
            if not mt5.symbol_select(symbol, True):
                error_code = mt5.last_error()
                logger.warning(f"Symbol select failed for {symbol}: Error code {error_code}")
                
            # Get symbol info to log properties
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                logger.debug(f"Symbol {symbol} properties: trade_mode={symbol_info.trade_mode}, " 
                           f"visible={symbol_info.visible}, " 
                           f"session_deals={getattr(symbol_info, 'session_deals', 'N/A')}, "
                           f"currency_base={symbol_info.currency_base}, "
                           f"currency_profit={symbol_info.currency_profit}")
            else:
                logger.warning(f"Could not get symbol info for {symbol}")
            
            # Fetch the data
            result = self.get_market_data(symbol, timeframe, num_candles)
            
            # Log the actual amount of data received
            if result is not None:
                actual_candles = len(result)
                logger.debug(f"Received {actual_candles}/{num_candles} requested candles for {symbol} on {timeframe}")
                if actual_candles < 100:
                    logger.warning(f"Insufficient data for {symbol} on {timeframe}: Only {actual_candles} candles available")
                    # Check last error from MT5
                    error_info = mt5.last_error()
                    if error_info[0] != 0:
                        logger.warning(f"MT5 error info when fetching {symbol} {timeframe}: {error_info}")
            else:
                logger.warning(f"No data received for {symbol} on {timeframe}")
                error_info = mt5.last_error()
                if error_info[0] != 0:
                    logger.warning(f"MT5 error info when fetching {symbol} {timeframe}: {error_info}")
                
            return result
        except Exception as e:
            logger.error(f"Error getting rates for {symbol} {timeframe}: {str(e)}")
            return None

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
        
    def get_min_stop_distance(self, symbol: str) -> Optional[float]:
        """Calculate and return the minimum stop distance for a symbol based on its current market conditions."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                # If the symbol info has a stops_level, use it multiplied by point
                if hasattr(symbol_info, "stops_level") and symbol_info.stops_level > 0:
                    return symbol_info.stops_level * symbol_info.point
                # Fallback: use 0.1% of the current ask price
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    return tick.ask * 0.001
            return None
        except Exception as e:
            logger.error(f"Error calculating min_stop_distance for {symbol}: {str(e)}")
            return None
            
    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if a symbol is available in MT5.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            bool: True if the symbol is available, False otherwise
        """
        try:
            if not self.connected:
                logger.warning("MT5 not connected when checking symbol availability")
                return False
                
            # Try to get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                # Try to select the symbol and check again
                selected = mt5.symbol_select(symbol, True)
                if not selected:
                    error_code = mt5.last_error()
                    logger.debug(f"Symbol {symbol} could not be selected: Error {error_code}")
                    return False
                    
                # Check again after selecting
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    return False
            
            # Check if symbol is enabled for trading
            if hasattr(symbol_info, 'trade_mode'):
                if symbol_info.trade_mode == 0:  # SYMBOL_TRADE_MODE_DISABLED
                    logger.debug(f"Symbol {symbol} is disabled for trading")
                    return False
            
            # Check if symbol has price data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.debug(f"No tick data available for {symbol}")
                return False
                
            # Check if bid/ask are valid
            if tick.bid <= 0 or tick.ask <= 0:
                logger.debug(f"Invalid prices for {symbol}: Bid={tick.bid}, Ask={tick.ask}")
                return False
                
            return True
                
        except Exception as e:
            logger.error(f"Error checking symbol availability for {symbol}: {str(e)}")
            return False
        
    def execute_trade(self, trade_params: Dict[str, Any]) -> Optional[List[int]]:
        """
        Execute trade on MT5 with partial take profits.
        
        Args:
            trade_params: Dictionary containing:
                - symbol: Trading symbol
                - signal_type: 'BUY' or 'SELL'
                - entry_price: Entry price
                - stop_loss: Stop loss price
                - position_size: Total position size
                - partial_tp_levels: List of dicts with 'ratio' and 'size' for each TP
                
        Returns:
            List of ticket numbers for opened positions, or None if execution failed
        """
        try:
            # Validate input parameters
            required_params = ['symbol', 'signal_type', 'entry_price', 'stop_loss', 
                             'position_size', 'partial_tp_levels']
            if not all(param in trade_params for param in required_params):
                logger.error(f"Missing required trade parameters. Required: {required_params}")
                return None
            
            # Calculate base risk
            risk = abs(trade_params['entry_price'] - trade_params['stop_loss'])
            base_volume = trade_params['position_size']
            orders = []
            
            # Process each partial take profit level
            for i, tp_level in enumerate(trade_params['partial_tp_levels']):
                # Calculate take profit price based on R-multiple
                if trade_params['signal_type'] == "BUY":
                    tp_price = trade_params['entry_price'] + (risk * tp_level['ratio'])
                else:  # SELL
                    tp_price = trade_params['entry_price'] - (risk * tp_level['ratio'])
                
                # Calculate volume for this partial
                partial_volume = base_volume * tp_level['size']
                if i == len(trade_params['partial_tp_levels']) - 1:
                    # Adjust last partial to account for any rounding errors
                    partial_volume = base_volume - sum(order['volume'] for order in orders)
                
                # Round volume to valid lot size
                symbol_info = mt5.symbol_info(trade_params['symbol'])
                if not symbol_info:
                    logger.error(f"Failed to get symbol info for {trade_params['symbol']}")
                    return None
                    
                lot_step = symbol_info.volume_step
                partial_volume = round(partial_volume / lot_step) * lot_step
                
                if partial_volume > 0:  # Only create order if volume is positive
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": trade_params['symbol'],
                        "volume": partial_volume,
                        "type": mt5.ORDER_TYPE_BUY if trade_params['signal_type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                        "price": trade_params['entry_price'],
                        "sl": trade_params['stop_loss'],
                        "tp": tp_price,
                        "deviation": 10,
                        "magic": 234000,
                        "comment": f"Python Bot - {trade_params['signal_type']} TP{i+1} ({tp_level['ratio']:.1f}R)",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    orders.append(request)
            
            # Ensure the symbol is selected before executing orders
            if not mt5.symbol_select(trade_params['symbol'], True):
                logger.error(f"Failed to select symbol: {trade_params['symbol']}")
                return None

            # Wait for fresh tick data
            retry_count = 0
            last_tick = mt5.symbol_info_tick(trade_params['symbol'])
            while not last_tick and retry_count < 3:
                time.sleep(0.5)
                last_tick = mt5.symbol_info_tick(trade_params['symbol'])
                retry_count += 1
            if not last_tick:
                logger.error("No tick data available for trade execution after retrying")
                return None

            # Update orders with current tick prices
            for order in orders:
                if order["type"] == mt5.ORDER_TYPE_BUY:
                    order["price"] = last_tick.ask
                else:
                    order["price"] = last_tick.bid
            
            # Execute all orders
            results = []
            for order in orders:
                result = mt5.order_send(order)
                if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                    logger.warning(f"Order requote detected: {result.comment}. Retrying with increased deviation.")
                    if "No prices" in result.comment:
                        tick = mt5.symbol_info_tick(order["symbol"])
                        if not tick:
                            logger.error("No tick data available to update order price")
                            raise Exception("No tick data available")
                        if order["type"] == mt5.ORDER_TYPE_BUY:
                            order["price"] = tick.ask
                        else:
                            order["price"] = tick.bid
                        logger.info(f"Updated order price to current market price: {order['price']}")
                    order["deviation"] += 10
                    result_retry = mt5.order_send(order)
                    if result_retry.retcode != mt5.TRADE_RETCODE_DONE:
                        raise Exception(f"Order failed after retry: {result_retry.comment}")
                    result = result_retry
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    raise Exception(f"Order failed: {result.comment}")
                
                results.append(result)
            
            logger.info(
                f"Successfully opened {len(results)} partial positions for {trade_params['symbol']} {trade_params['signal_type']}\n" +
                "\n".join([f"  Partial {i+1}: {order['volume']:.2f} lots, TP at {order['tp']:.5f} ({tp_level['ratio']:.1f}R)"
                          for i, (order, tp_level) in enumerate(zip(orders, trade_params['partial_tp_levels']))])
            )
            
            return [result.order for result in results]
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

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
            current_balance = self.get_account_info().get("balance", 0)
            
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
        current_balance = self.get_account_info().get("balance", 0)
        
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

    def open_buy(self, symbol: str, volume: float, stop_loss: float = 0.0, 
                take_profit: float = 0.0, comment: str = "") -> Optional[int]:
        """
        Open a buy position for the specified symbol.
        
        Args:
            symbol: Trading instrument symbol
            volume: Trade volume in lots
            stop_loss: Stop loss level
            take_profit: Take profit level
            comment: Order comment
            
        Returns:
            Position ticket on success, None on failure
        """
        # Get the current ask price for buying
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found")
            return None
            
        price = symbol_info.ask
        
        # Execute buy order using helper method
        return self._execute_order(
            symbol=symbol,
            volume=volume,
            action=mt5.ORDER_TYPE_BUY,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )
        
    def open_sell(self, symbol: str, volume: float, stop_loss: float = 0.0, 
                take_profit: float = 0.0, comment: str = "") -> Optional[int]:
        """
        Open a sell position for the specified symbol.
        
        Args:
            symbol: Trading instrument symbol
            volume: Trade volume in lots
            stop_loss: Stop loss level
            take_profit: Take profit level
            comment: Order comment
            
        Returns:
            Position ticket on success, None on failure
        """
        # Get the current bid price for selling
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found")
            return None
            
        price = symbol_info.bid
        
        # Execute sell order using helper method
        return self._execute_order(
            symbol=symbol,
            volume=volume,
            action=mt5.ORDER_TYPE_SELL,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )

    def get_last_error(self) -> Optional[str]:
        """
        Get the last MT5 error message.
        
        Returns:
            Optional[str]: The last error message from MT5 or None if no error
        """
        try:
            # Initialize class attribute if needed
            if not hasattr(self, '_last_error'):
                self._last_error = None
                
            # Get the last error from MT5
            mt5_error = mt5.last_error()
            
            # Format depends on whether it's a tuple or a single value
            if isinstance(mt5_error, tuple) and len(mt5_error) >= 2:
                error_code, error_description = mt5_error[0], mt5_error[1]
                
                # Only create error message for actual errors (code != 0)
                if error_code != 0:
                    self._last_error = f"MT5 Error: {error_code} - {error_description}"
                    return self._last_error
                else:
                    # This is a "success" message, not an error
                    return None
            elif isinstance(mt5_error, int):
                # Handle case where only error code is returned
                if mt5_error != 0:
                    self._last_error = f"MT5 Error Code: {mt5_error}"
                    return self._last_error
                else:
                    return None
            else:
                # For any other format, return as string
                if mt5_error:
                    self._last_error = f"MT5 Error: {mt5_error}"
                    return self._last_error
            
            return self._last_error
        except Exception as e:
            logger.error(f"Error getting MT5 last error: {str(e)}")
            return f"Error retrieving MT5 error: {str(e)}"
        
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

    def calculate_max_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate maximum allowed position size based on available margin.
        
        Args:
            symbol: Trading symbol
            price: Current price
            
        Returns:
            float: Maximum allowed position size in lots
        """
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return 0.0
            
            # Use a realistic leverage instead of potentially glitched values
            leverage = min(account_info.leverage, 500)
            if leverage <= 0 or leverage > 500:
                leverage = 100  # Default to 1:100 leverage if unrealistic value detected
                
            # Log account details for debugging
            logger.info(f"Account details - Balance: {account_info.balance:.2f}, Free Margin: {account_info.margin_free:.2f}, Leverage: {leverage}")
            
            # Ensure symbol is visible in MarketWatch
            if not symbol_info.visible:
                logger.warning(f"{symbol} not visible, trying to add it")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to add {symbol} to MarketWatch")
                    return 0.0
            
            # Try to get contract size and calculate basic margin
            contract_size = symbol_info.trade_contract_size
            
            # Fallback approach if MT5 margin calculation fails
            try:
                # Try MT5's built-in margin calculation first
                margin_1_lot = mt5.order_calc_margin(
                    mt5.ORDER_TYPE_BUY,  # Direction doesn't matter for margin calculation
                    symbol,
                    1.0,    # 1 lot
                    price
                )
                
                if margin_1_lot is None or margin_1_lot == 0:
                    # MT5 margin calculation failed, use a fallback approach
                    logger.warning(f"MT5 margin calculation failed for {symbol}, using fallback calculation")
                    
                    # Special handling for JPY pairs which have different pip values
                    if "JPY" in symbol:
                        # For JPY pairs, use standard margin calculation but with proper scaling
                        logger.info(f"Using JPY-specific margin calculation for {symbol}")
                        # For JPY pairs: (price * contract_size) / leverage
                        margin_1_lot = (price * contract_size) / leverage
                    else:
                        # Standard margin calculation for non-JPY pairs
                        margin_1_lot = (price * contract_size) / leverage
                    
                    logger.info(f"Fallback margin calculation: Price={price}, ContractSize={contract_size}, Leverage={leverage}")
                else:
                    logger.info(f"Margin required for 1 lot of {symbol}: {margin_1_lot:.2f}")
            except Exception as e:
                logger.error(f"Both primary and fallback margin calculations failed: {str(e)}")
                return 0.0
                
            # Calculate maximum lots based on available margin
            available_margin = account_info.margin_free
            
            # Use only a portion of free margin (50%) as a safety measure - reduced from 90%
            max_lots = (available_margin * 0.5) / margin_1_lot
            
            # Cap max lots to a reasonable amount based on account size
            reasonable_max = account_info.balance / 1000  # $1000 of account balance = 1 lot max
            max_lots = min(max_lots, reasonable_max)
            
            # Round down to symbol minimum lot step
            max_lots = math.floor(max_lots / symbol_info.volume_step) * symbol_info.volume_step
            
            # Ensure within symbol limits
            max_lots = min(max_lots, symbol_info.volume_max)
            max_lots = max(0.0, max_lots)  # Ensure non-negative
            
            logger.info(f"Maximum allowed position size for {symbol}: {max_lots:.4f} lots")
            logger.info(f"Available margin: {available_margin:.2f}, Margin per lot: {margin_1_lot:.2f}")
            
            return max_lots
            
        except Exception as e:
            logger.error(f"Error calculating max position size: {str(e)}")
            # Add traceback for better debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0

    def adjust_position_size(self, symbol: str, requested_size: float, price: float) -> float:
        """
        Adjust requested position size to fit within available margin.
        
        Args:
            symbol: Trading symbol
            requested_size: Requested position size in lots
            price: Current price
            
        Returns:
            float: Adjusted position size that fits within available margin
        """
        # Get symbol info for minimum lot size
        symbol_info = mt5.symbol_info(symbol)  # type: ignore
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return 0.0
            
        min_lot = symbol_info.volume_min
        volume_step = symbol_info.volume_step
        
        logger.debug(f"Symbol {symbol} - Min lot: {min_lot}, Volume step: {volume_step}")
            
        # Calculate maximum position size based on available margin
        max_size = self.calculate_max_position_size(symbol, price)
        
        # If no margin available, log this specifically
        if max_size <= 0:
            logger.warning(f"No margin available for {symbol}. Consider adding funds or using smaller position sizes.")
            return 0.0
            
        # If max size is less than minimum lot size, try to use exactly minimum lot size
        # but only if we have at least 80% of the required margin
        if max_size < min_lot:
            # Get account info for margin check
            account_info = mt5.account_info()  # type: ignore
            if account_info:
                # Calculate margin required for minimum lot size
                try:
                    margin_min_lot = mt5.order_calc_margin(
                        mt5.ORDER_TYPE_BUY,
                        symbol,
                        min_lot,
                        price
                    )
                    
                    # Fallback calculation if MT5 margin calculation fails
                    if margin_min_lot is None or margin_min_lot == 0:
                        contract_size = symbol_info.trade_contract_size
                        leverage = account_info.leverage
                        
                        # Special handling for JPY pairs which have different pip values
                        if "JPY" in symbol:
                            logger.info(f"Using JPY-specific margin calculation for {symbol}")
                            margin_min_lot = (price * contract_size * min_lot) / leverage
                        else:
                            margin_min_lot = (price * contract_size * min_lot) / leverage
                        
                        logger.info(f"Fallback margin calculation: Price={price}, ContractSize={contract_size}, Leverage={leverage}, MinLot={min_lot}")
                    
                    # If we have at least 80% of required margin, allow minimum lot size
                    if account_info.margin_free >= margin_min_lot * 0.8:
                        logger.warning(f"Available margin only allows {max_size:.4f} lots, but using minimum lot size {min_lot} for {symbol}")
                        return min_lot
                except Exception as e:
                    logger.error(f"Error calculating margin for minimum lot size: {str(e)}")
            
            logger.warning(f"Insufficient margin for minimum lot size ({min_lot}) for {symbol}")
            return 0.0
            
        # Normal adjustment: use either requested size or max size, whichever is smaller
        adjusted_size = min(requested_size, max_size)
        
        # Round to the nearest valid lot size based on volume_step
        if volume_step > 0:
            steps = round(adjusted_size / volume_step)
            adjusted_size = steps * volume_step
            
            # Log the rounding adjustment if significant
            original = min(requested_size, max_size)
            if abs(original - adjusted_size) > volume_step / 2:
                logger.debug(f"Rounded position size from {original:.4f} to {adjusted_size:.4f} lots to match volume step")
        
        # Make sure it's not below minimum lot size
        if adjusted_size < min_lot:
            # For crypto pairs like ETH and BTC, check if we're close to a valid subdivision
            # Many brokers allow 0.01 for ETH/BTC even if they report min_lot as 0.1
            if ('ETH' in symbol or 'BTC' in symbol) and adjusted_size >= 0.01:
                # Round to nearest valid lot size using volume_step
                if volume_step > 0:
                    steps = round(adjusted_size / volume_step)
                    adjusted_size = steps * volume_step
                    adjusted_size = max(volume_step, adjusted_size)  # Ensure minimum of one step
                else:
                    # Fallback to old method if volume_step is 0
                    adjusted_size = round(adjusted_size * 100) / 100
                
                # Ensure we're not below the minimum lot size
                adjusted_size = max(min_lot, adjusted_size)
                
                logger.info(f"Adjusted crypto position size to {adjusted_size:.4f} lots (step: {volume_step})")
            else:
                # For other instruments, either use min_lot or 0 depending on how close we are
                if adjusted_size >= min_lot * 0.8:
                    adjusted_size = min_lot
                    logger.info(f"Rounded position size up to minimum lot size {min_lot}")
                else:
                    adjusted_size = 0.0
                    logger.warning(f"Adjusted size {adjusted_size:.4f} is too small compared to minimum lot size {min_lot}. Setting to 0.")
        
        if adjusted_size < requested_size:
            logger.warning(f"Reduced position size from {requested_size:.4f} to {adjusted_size:.4f} lots due to margin constraints")
            
        return adjusted_size

    def create_trade_request(self, symbol: str, volume: float, action: int,
                          price: float, stop_loss: float = 0.0, 
                          take_profit: float = 0.0, comment: str = "") -> Dict[str, Any]:
        """
        Create a trade request dictionary for MT5.
        
        Args:
            symbol: Trading instrument symbol
            volume: Trade volume in lots
            action: Trade action (MT5 constant)
            price: Order price
            stop_loss: Stop loss level
            take_profit: Take profit level
            comment: Order comment
            
        Returns:
            Dictionary with trade request parameters
        """
        # Import locally to avoid circular imports
        import MetaTrader5 as mt5
        
        # Get symbol info for proper formatting
        symbol_info = self.get_symbol_info(symbol)
        
        # Get current account information
        account_info = self.get_account_info()
        
        # Safety checks on inputs
        if symbol_info is None:
            logger.warning(f"Symbol info not available for {symbol} when creating trade request")
            return {}
            
        # Make sure we have a valid login from account_info
        login = account_info.get("login") if account_info else 0
        # Fallback to direct MT5 call if needed
        if not login and mt5.account_info():
            login = mt5.account_info().login
        
        # Sanitize the comment - MT5 has strict requirements for comments
        if comment:
            # Only allow alphanumeric and spaces - no special characters at all
            sanitized_comment = re.sub(r'[^a-zA-Z0-9 ]', '', comment)
            # Limit length to 20 characters (MT5 is very strict with comments)
            sanitized_comment = sanitized_comment[:20]
            # If comment is empty after sanitization, use a default
            if not sanitized_comment.strip():
                sanitized_comment = "MT5Trade"
        else:
            sanitized_comment = "MT5Trade"
        
        # Ensure proper float formatting for numeric values
        try:
            # Get proper price precision from symbol_info
            digits = getattr(symbol_info, "digits", 5)
            
            # Round price according to symbol digits
            price = round(float(price), digits) if price is not None else 0.0
            
            # Process stop loss and take profit
            if stop_loss is not None and stop_loss > 0:
                # For sell orders, stop loss must be above entry price
                if action == mt5.ORDER_TYPE_SELL and stop_loss <= price:
                    stop_loss = price + (price * 0.01)  # Default to 1% above price
                # For buy orders, stop loss must be below entry price
                elif action == mt5.ORDER_TYPE_BUY and stop_loss >= price:
                    stop_loss = price - (price * 0.01)  # Default to 1% below price
                stop_loss = round(float(stop_loss), digits)
            else:
                stop_loss = 0.0
                
            if take_profit is not None and take_profit > 0:
                # For buy orders, take profit must be above entry price
                if action == mt5.ORDER_TYPE_BUY and take_profit <= price:
                    take_profit = price + (price * 0.02)  # Default to 2% above price
                # For sell orders, take profit must be below entry price
                elif action == mt5.ORDER_TYPE_SELL and take_profit >= price:
                    take_profit = price - (price * 0.02)  # Default to 2% below price
                take_profit = round(float(take_profit), digits)
            else:
                take_profit = 0.0
                
            # Make sure volume is properly formatted
            volume = round(float(volume), 2) if volume is not None else 0.01
        except (TypeError, ValueError) as e:
            logger.error(f"Error formatting numeric values for trade request: {str(e)}")
            # Use safe defaults
            price = float(price) if price is not None else 0.0
            stop_loss = float(stop_loss) if stop_loss is not None else 0.0
            take_profit = float(take_profit) if take_profit is not None else 0.0
            volume = float(volume) if volume is not None else 0.01
        
        # Create the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": action,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": sanitized_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "login": login
        }
        
        logger.debug(f"Created trade request: {request}")
        return request

    def _execute_order(self, symbol: str, volume: float, action: int, 
                    price: float, stop_loss: float = 0.0, 
                    take_profit: float = 0.0, comment: str = "") -> Optional[int]:
        """
        Execute a trade order with risk checks.
        
        Args:
            symbol: Trading instrument symbol
            volume: Trade volume in lots
            action: Trade action (MT5 constant)
            price: Order price
            stop_loss: Stop loss level
            take_profit: Take profit level
            comment: Order comment
            
        Returns:
            Position ticket on success, None on failure
        """
        # Check connection
        if not self.is_connected():
            logger.error("Cannot execute order - not connected to MT5")
            return None
            
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found")
            return None
            
        # Adjust volume based on available margin
        account_info = self.get_account_info()
        if not account_info:
            logger.error("Cannot retrieve account info")
            return None
            
        # Log detailed account info for debugging
        logger.info(f"[EXECUTE] Account balance: {account_info.get('balance', 0)}, Equity: {account_info.get('equity', 0)}, Margin: {account_info.get('margin', 0)}, Free Margin: {account_info.get('margin_free', 0)}")
        
        # Calculate adjusted volume based on free margin
        free_margin = account_info.get('margin_free', 0)
        margin_for_one_lot = getattr(symbol_info, 'margin_initial', 0)
        
        # Ensure we have a valid margin calculation
        if margin_for_one_lot <= 0:
            # Use a fallback calculation based on leverage if available
            leverage = account_info.get('leverage', 100)  # Default to 100 if not found
            # Estimate margin per lot based on symbol price and leverage
            symbol_price = getattr(symbol_info, 'ask', 0) or getattr(symbol_info, 'bid', 0) or price
            # Standard lot size is 100,000 units
            standard_lot_value = 100000 * symbol_price
            margin_for_one_lot = standard_lot_value / leverage if leverage > 0 else standard_lot_value / 100
            
            logger.info(f"[EXECUTE] Using fallback margin calculation: price={symbol_price}, leverage={leverage}, margin_for_one_lot={margin_for_one_lot}")
        
        # Ensure margin_for_one_lot is at least a small positive value to prevent division by zero
        margin_for_one_lot = max(margin_for_one_lot, 0.01)
        
        max_possible_volume = free_margin / margin_for_one_lot * 0.9 if margin_for_one_lot > 0 else volume  # Use 90% of free margin max
        
        # Special handling for high-value instruments like gold
        if symbol.startswith("XAU") or symbol.startswith("GOLD"):
            # For gold, adjust the calculation to account for its high value
            # Instead of standard lot size, use 100 units for gold (standard is usually 100 oz)
            adjusted_margin = margin_for_one_lot / 1000  # Reduce margin requirement by factor
            max_possible_volume = free_margin / adjusted_margin * 0.9
            logger.info(f"[EXECUTE] Special handling for gold: adjusted margin={adjusted_margin}, max_volume={max_possible_volume}")
        # Special handling for JPY pairs which have different pricing scales
        elif "JPY" in symbol:
            # For JPY pairs, adjust the margin requirement to account for their different pricing scale
            adjusted_margin = margin_for_one_lot / 100  # Reduce margin requirement for JPY pairs
            max_possible_volume = free_margin / adjusted_margin * 0.9
            logger.info(f"[EXECUTE] Special handling for JPY pair: adjusted margin={adjusted_margin}, max_volume={max_possible_volume}")
        
        # Ensure we can place at least the minimum order size if we have sufficient free margin
        min_volume = getattr(symbol_info, 'volume_min', 0.01)
        
        # Force minimum volume for specific cases when we have reasonable margin
        if free_margin > 50:  # If we have at least $50 of free margin
            if ((symbol.startswith("XAU") or symbol.startswith("GOLD")) and max_possible_volume < min_volume) or \
               ("JPY" in symbol and max_possible_volume < min_volume):
                # Force minimum volume for gold and JPY pairs with reasonable margin
                max_possible_volume = min_volume
                logger.info(f"[EXECUTE] Forcing minimum volume {min_volume} for {symbol} with free margin {free_margin}")
        
        if free_margin > min_volume * margin_for_one_lot and max_possible_volume < min_volume:
            max_possible_volume = min_volume
            logger.info(f"[EXECUTE] Setting max_possible_volume to minimum allowed: {min_volume}")
        
        logger.info(f"[EXECUTE] Volume calculation - Free margin: {free_margin}, Margin per lot: {margin_for_one_lot}, Max possible volume: {max_possible_volume}, Requested volume: {volume}")
        
        adjusted_volume = min(volume, max_possible_volume)
        
        # For high-value instruments, ensure minimum volume
        if (symbol.startswith("XAU") or symbol.startswith("GOLD")) and adjusted_volume < min_volume and free_margin > 50:
            # If we have reasonable free margin, force minimum volume for gold
            adjusted_volume = min_volume
            logger.info(f"[EXECUTE] Forcing minimum volume {min_volume} for gold with free margin {free_margin}")
        
        # Ensure we don't go below minimum volume if we have enough margin
        if adjusted_volume < min_volume and free_margin >= min_volume * margin_for_one_lot:
            adjusted_volume = min_volume
            logger.info(f"[EXECUTE] Adjusted volume to minimum allowed: {min_volume}")
        
        # If volume was adjusted, log the reason
        if adjusted_volume < volume:
            logger.warning(f"Position size adjusted from {volume:.4f} to {adjusted_volume:.4f} lots due to margin constraints")
        
        # Round volume to valid step size
        volume_step = getattr(symbol_info, 'volume_step', 0.01)
        adjusted_volume = round(adjusted_volume / volume_step) * volume_step
        
        logger.info(f"[EXECUTE] Final volume after rounding: {adjusted_volume}, Volume step: {volume_step}")
        
        # Validate if volume is within allowed range
        min_volume = getattr(symbol_info, 'volume_min', 0.01)
        max_volume = getattr(symbol_info, 'volume_max', 100.0)
        
        logger.info(f"[EXECUTE] Volume validation - Min: {min_volume}, Max: {max_volume}, Final: {adjusted_volume}")
        
        if adjusted_volume < min_volume:
            logger.error(f"Volume {adjusted_volume} is below minimum allowed {min_volume}")
            return None
            
        if adjusted_volume > max_volume:
            logger.error(f"Volume {adjusted_volume} is above maximum allowed {max_volume}")
            return None
        
        # Prepare the request using the helper method
        request = self.create_trade_request(
            symbol=symbol,
            volume=adjusted_volume,
            action=action,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )
        
        logger.info(f"[EXECUTE] Order request: {request}")
            
        # Execute the order
        try:
            logger.info(f"[EXECUTE] Sending order to MT5: {symbol} {'BUY' if action == mt5.ORDER_TYPE_BUY else 'SELL'} {adjusted_volume} lots at {price}")
            result = mt5.order_send(request)  # type: ignore
            
            if result is None:
                error = mt5.last_error()  # type: ignore
                logger.error(f"[EXECUTE_FAILURE] MT5 returned None. Error: {error}")
                return None
                
            logger.info(f"[EXECUTE] Order result: retcode={result.retcode}, comment={result.comment}")
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[EXECUTE_SUCCESS] Trade executed: {symbol} {'BUY' if action == mt5.ORDER_TYPE_BUY else 'SELL'} {adjusted_volume} lots, ticket #{result.order}")
                return result.order
            else:
                error_code = result.retcode if result else -1
                error_desc = self._get_error_description(error_code)
                error_message = mt5.last_error()  # type: ignore
                logger.error(f"[EXECUTE_FAILURE] Failed to execute trade: {error_code} - {error_desc}. MT5 error: {error_message}")
                logger.error(f"[EXECUTE_FAILURE] Request details: {request}")
                return None
        except Exception as e:
            logger.error(f"[EXECUTE_FAILURE] Exception during order_send: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _get_error_description(self, error_code: int) -> str:
        """
        Get a descriptive message for MT5 error codes.
        
        Args:
            error_code: MT5 error code
            
        Returns:
            Description of the error
        """
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
            10019: "Request accepted for execution",
            10020: "Request rejected for reprocessing",
            10021: "Request completed, and the result is unknown",
            10022: "Request completed, and the result is known",
            10023: "Request rejected by processing timeout",
            10024: "Request rejected due to the filled order's expiration",
            10025: "Request placed in the processing queue",
            10026: "Request accepted for processing, but the execution is rejected",
            10027: "Request accepted for processing with unknown result",
            10028: "Request rejected for processing due to money issues",
            10029: "Request rejected for processing due to position opening issues",
            
            # Account errors
            10051: "Invalid account",
            10052: "Invalid trade account",
            10053: "Account disabled",
            10054: "Too many connected clients",
            10055: "Too many requests",
            10056: "AutoTrading disabled for the account",
            10057: "AutoTrading disabled for the server",
            10058: "AutoTrading disabled for the symbol",
            
            # Trade errors
            10130: "Invalid volume",
            10131: "Invalid price",
            10132: "Invalid stops",
            10133: "Trade is disabled",
            10134: "Not enough money",
            10135: "Price changed",
            10136: "No prices",
            10137: "Invalid expiration",
            10138: "Order state changed",
            10139: "Too many orders",
            10140: "Too many position or orders for symbol",
            10141: "Position exists already",
            10142: "Position does not exist",
            10143: "Hedge positions are prohibited",
            10144: "Position close prohibited",
            10145: "Positions or orders are prohibited",
            10146: "Invalid symbol/pair",
            10147: "Invalid price for StopLoss",
            10148: "Invalid price for TakeProfit",
            
            # Common errors
            -1: "Unknown error",
            -2: "No connection",
            -3: "Not enough rights",
            -4: "Too frequent requests",
            -5: "Timeout",
            -6: "Invalid parameter",
            -7: "Prohibited by FIFO rule",
            -8: "Not connected",
            -9: "No prices",
            -10: "Invalid trading mode",
            -11: "Not initialized",
            -12: "Platform busy",
            -13: "Critical error",
            -14: "Server disconnected",
            -15: "Some error",
            -16: "Unknown error",
            -17: "Invalid handle",
            -18: "Locked operation",
            -19: "Resources unavailable",
            -20: "Not enough memory",
            -21: "Cannot open file",
            -22: "Cannot write file",
            -23: "Cannot read file",
            -24: "Invalid date",
            -25: "Internal error"
        }
        
        return error_descriptions.get(error_code, f"Unknown error code: {error_code}")

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
