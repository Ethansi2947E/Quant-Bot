# mypy: ignore-errors
# pyright: reportAttributeAccessIssue=false
# flake8: noqa

import MetaTrader5 as mt5  # type: ignore
from datetime import datetime, timedelta, UTC
import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any, cast
import time
import traceback
import json
import math
import sys
import re
import asyncio
import numpy as np

from config.config import MT5_CONFIG, TRADING_CONFIG

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

# Real-time market data monitoring
class RealTimeDataCallback:
    """Callback class for real-time data from MT5"""
    
    def __init__(self, symbol, timeframe, callback_function):
        """
        Initialize a real-time data callback for MT5.
        
        Args:
            symbol: Trading symbol to monitor
            timeframe: Timeframe to monitor
            callback_function: Function to call with processed data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.callback_function = callback_function
        
    async def on_tick(self, tick_data):
        """
        Process tick data from MT5 and pass it to the callback function.
        
        Args:
            tick_data: Raw tick data from MT5
        """
        try:
            # Access MT5Handler to process the tick
            mt5_handler = MT5Handler.get_instance()
            if not mt5_handler:
                logger.error("MT5Handler instance not available for tick processing")
                return
            
            # Create an enhanced tick object with symbol information
            enhanced_tick_data = {'original_tick': tick_data, 'symbol': self.symbol}
            
            # Process the tick data
            processed_tick = await mt5_handler.on_tick(enhanced_tick_data)
            
            if processed_tick:
                # Call the callback function with the processed data
                await self.callback_function(
                    self.symbol, 
                    self.timeframe, 
                    processed_tick, 
                    'tick'
                )
        except Exception as e:
            logger.error(f"Error in on_tick callback for {self.symbol}: {e}")
            
    async def on_new_candle(self, candle_data):
        """
        Process candle data from MT5 and pass it to the callback function.
        
        Args:
            candle_data: Raw candle data from MT5
        """
        try:
            # Access MT5Handler to process the candle
            mt5_handler = MT5Handler.get_instance()
            if not mt5_handler:
                logger.error("MT5Handler instance not available for candle processing")
                return
            
            # Process the candle data
            processed_candles = await mt5_handler.on_candle(candle_data, self.symbol, self.timeframe)
            
            if processed_candles:
                # Call the callback function with the processed data
                await self.callback_function(
                    self.symbol, 
                    self.timeframe, 
                    processed_candles, 
                    'candle'
                )
        except Exception as e:
            logger.error(f"Error in on_new_candle callback for {self.symbol}/{self.timeframe}: {e}")
            logger.exception(e)

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
        
        # Add real-time data monitoring
        self.real_time_callbacks = {}
        self.real_time_enabled = False
        self.real_time_monitoring_task = None
    
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
        try:
            # Get the current ask price for buying
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return None
                
            entry_price = symbol_info.ask
            
            # Format parameters for execute_trade
            trade_params = {
                'symbol': symbol,
                'signal_type': 'BUY',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'position_size': volume,
                'partial_tp_levels': [{'ratio': 1.0, 'size': 1.0}]  # Single TP level
            }
            
            # If take profit is provided, set it
            if take_profit > 0:
                trade_params['partial_tp_levels'][0]['ratio'] = abs(take_profit - entry_price) / abs(entry_price - stop_loss) if stop_loss > 0 else 1.0
            
            # Execute the trade
            logger.debug(f"Executing BUY trade for {symbol} with volume {volume}, SL {stop_loss}, TP {take_profit}")
            result = self.execute_trade(trade_params)
            
            # Return the first ticket if successful
            if result and len(result) > 0:
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Error in open_buy: {str(e)}")
            return None

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
        try:
            # Get the current bid price for selling
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return None
                
            entry_price = symbol_info.bid
            
            # Format parameters for execute_trade
            trade_params = {
                'symbol': symbol,
                'signal_type': 'SELL',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'position_size': volume,
                'partial_tp_levels': [{'ratio': 1.0, 'size': 1.0}]  # Single TP level
            }
            
            # If take profit is provided, set it
            if take_profit > 0:
                trade_params['partial_tp_levels'][0]['ratio'] = abs(entry_price - take_profit) / abs(stop_loss - entry_price) if stop_loss > 0 else 1.0
            
            # Execute the trade
            logger.debug(f"Executing SELL trade for {symbol} with volume {volume}, SL {stop_loss}, TP {take_profit}")
            result = self.execute_trade(trade_params)
            
            # Return the first ticket if successful
            if result and len(result) > 0:
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Error in open_sell: {str(e)}")
            return None

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
                    if error_code in error_descriptions:
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

    def calculate_position_size(self, symbol: str, price: float, 
                                risk_amount: Optional[float] = None, 
                                risk_percent: Optional[float] = None,
                                entry_price: Optional[float] = None,
                                stop_loss_price: Optional[float] = None,
                                max_percent_of_balance: float = 3.0,
                                enforce_limits: bool = True) -> float:
   
        try:
            # Get symbol info and account info
            symbol_info = self.get_symbol_info(symbol)
            account_info = self.get_account_info()
            
            if not symbol_info or not account_info:
                logger.error(f"Cannot calculate position size - missing symbol info or account info")
                return 0.0
                
            # Extract needed values
            account_balance = account_info.get('balance', 0.0)
            free_margin = account_info.get('margin_free', 0.0)
            leverage = account_info.get('leverage', 100)
            
            # Get contract & price specifics
            contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
            digit_multiplier = 10 ** getattr(symbol_info, 'digits', 5)
            point_value = getattr(symbol_info, 'point', 0.00001)
            min_lot = getattr(symbol_info, 'volume_min', 0.01)
            max_lot = getattr(symbol_info, 'volume_max', 100.0)
            lot_step = getattr(symbol_info, 'volume_step', 0.01)
            
            # Get margin required per lot
            margin_initial = getattr(symbol_info, 'margin_initial', 0.0)
            
            # Fallback margin calculation if not available
            if margin_initial <= 0:
                standard_lot_value = contract_size * price
                margin_initial = standard_lot_value / leverage
            
            # Safety check to prevent division by zero
            margin_initial = max(margin_initial, 0.01) 
            
            # CALCULATION METHOD 1: Based on risk amount and stop loss
            if risk_amount is not None and entry_price is not None and stop_loss_price is not None:
                # Calculate position size based on fixed risk amount
                risk_amount = min(risk_amount, account_balance * max_percent_of_balance / 100)
                stop_distance = abs(entry_price - stop_loss_price)
                
                if stop_distance <= 0:
                    logger.error(f"Invalid stop distance for {symbol}: entry={entry_price}, stop={stop_loss_price}")
                    return 0.0
                    
                # Calculate appropriate position size in standard lots based on risk
                one_pip_value = (contract_size / digit_multiplier) * point_value * 10
                pips_at_risk = stop_distance / point_value
                position_size = risk_amount / (pips_at_risk * one_pip_value)
                
            # CALCULATION METHOD 2: Based on risk percentage and stop loss
            elif risk_percent is not None and entry_price is not None and stop_loss_price is not None:
                # Convert percentage to decimal
                risk_decimal = min(risk_percent / 100, max_percent_of_balance / 100)
                
                # Calculate risk amount
                risk_amount = account_balance * risk_decimal
                stop_distance = abs(entry_price - stop_loss_price)
                
                if stop_distance <= 0:
                    logger.error(f"Invalid stop distance for {symbol}: entry={entry_price}, stop={stop_loss_price}")
                    return 0.0
                    
                # Calculate position size based on risk percentage
                one_pip_value = (contract_size / digit_multiplier) * point_value * 10
                pips_at_risk = stop_distance / point_value
                position_size = risk_amount / (pips_at_risk * one_pip_value)
                
            # CALCULATION METHOD 3: Based on maximum affordable size
            else:
                # Calculate max position size based on free margin
                max_size_by_margin = free_margin / margin_initial * 0.9  # Use 90% of free margin
                position_size = max_size_by_margin
            
            # Apply symbol-specific adjustments if needed
            # Special handling for gold
            if symbol.startswith("XAU") or symbol.startswith("GOLD"):
                # Gold typically has higher margin requirements
                position_size *= 0.1  # Reduce position size for gold
            
            # Special handling for JPY pairs
            elif "JPY" in symbol:
                # JPY pairs have different pip values
                position_size *= 0.1  # Adjust for JPY pairs
            
            # Special handling for crypto
            elif any(crypto in symbol for crypto in ["BTC", "ETH", "LTC", "XRP"]):
                position_size *= 0.01  # Significant reduction for crypto due to volatility
            
            # ENFORCE SIZE LIMITS if requested
            if enforce_limits:
                # Apply min/max volume constraints
                position_size = max(min_lot, min(position_size, max_lot))
                
                # Round to valid lot step
                position_size = round(position_size / lot_step) * lot_step
                
                # Log any adjustments made
                logger.info(f"Position size calculated for {symbol}: {position_size} lots")
                
                # Double-check against free margin
                required_margin = position_size * margin_initial
                if required_margin > free_margin:
                    # Further reduce position size if needed
                    adjusted_size = (free_margin * 0.9) / margin_initial
                    adjusted_size = round(adjusted_size / lot_step) * lot_step
                    adjusted_size = max(min_lot, min(adjusted_size, max_lot))
                    
                    logger.warning(f"Position size adjusted from {position_size} to {adjusted_size} due to margin constraints")
                    position_size = adjusted_size
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Default to minimum size on error

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

    def start_real_time_monitoring(self, symbols=None, timeframes=None, callback_function=None):
        """
        Start real-time market data monitoring.
        
        Args:
            symbols: List of symbols to monitor
            timeframes: List of timeframes to monitor
            callback_function: Function to call with processed data
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        logger.info("🚀 Starting real-time market data monitoring")
        
        # Initialize the real-time monitoring dict if not already done
        if not hasattr(self, 'rt_monitoring'):
            self.rt_monitoring = {
                "running": False,
                "task": None,
                "callbacks": []
            }
            
        # Get default symbols and timeframes if not provided
        if not symbols:
            # Check if symbols in TRADING_CONFIG are in the new format (dict with symbol and timeframe)
            symbols_config = MT5_CONFIG.get("symbols", [])
            trading_symbols = TRADING_CONFIG.get("symbols", [])
            
            if trading_symbols and isinstance(trading_symbols[0], dict):
                # Extract just the symbol names from the dictionary format
                symbols = [item['symbol'] for item in trading_symbols]
            else:
                # Use the old format or default
                symbols = list(trading_symbols if trading_symbols else symbols_config)
                
            if not symbols:  # If still empty, use default
                symbols = ["EURUSD"]
            
            logger.info(f"📊 Using symbols from config: {symbols}")
            
        if not timeframes:
            # Get timeframes from TRADING_CONFIG
            timeframes = TRADING_CONFIG.get("timeframes", ["M15"])
            
            # Limit to just 1-2 timeframes to avoid excessive data
            if len(timeframes) > 2:
                logger.warning(f"⚠️ Too many timeframes configured: {timeframes}. Using only primary timeframes.")
                # Use only the first 2 timeframes (typically M1/M5 or M15)
                timeframes = timeframes[:2]
                
            logger.info(f"📈 Using timeframes from config: {timeframes}")
            
        # Check if MT5 is connected
        if not self.is_connected():
            logger.error("❌ MT5 is not connected, cannot start real-time monitoring")
            return False
            
        # Create callback objects for each symbol/timeframe combination
        for symbol in symbols:
            for timeframe in timeframes:
                callback = RealTimeDataCallback(symbol, timeframe, callback_function)
                self.rt_monitoring["callbacks"].append(callback)
                logger.info(f"✅ Added callback for {symbol}/{timeframe}")
                
        # Start the monitoring task if not already running
        if not self.rt_monitoring.get("running", False):
            self.rt_monitoring["running"] = True
            
            # Start the monitoring task
            try:
                self._start_monitoring_task()
                logger.info(f"🎉 Started real-time monitoring for {len(symbols)} symbols and {len(timeframes)} timeframes")
                return True
            except Exception as e:
                logger.error(f"❌ Error starting real-time monitoring: {str(e)}")
                self.rt_monitoring["running"] = False
                return False
        else:
            logger.info("ℹ️ Real-time monitoring is already running")
            return True

    def _start_monitoring_task(self):
        """Start the monitoring task in the background."""
        
        async def monitoring_loop():
            """Main monitoring loop for real-time data."""
            logger.info("🔄 Real-time monitoring loop started")
            
            while self.rt_monitoring.get("running", False):
                try:
                    # Check connection first
                    if not self.is_connected():
                        logger.warning("⚠️ MT5 connection lost during monitoring, attempting to reconnect")
                        if not await self._reconnect():
                            # If reconnection failed, sleep and try again
                            await asyncio.sleep(5)
                            continue
                    
                    # Get all callbacks
                    callbacks = self.rt_monitoring.get("callbacks", [])
                    
                    # Process each symbol - group callbacks by symbol to avoid duplicate tick processing
                    symbols_processed = set()
                    symbol_timeframe_map = {}
                    
                    # First organize callbacks by symbol and timeframe
                    for callback in callbacks:
                        symbol = callback.symbol
                        timeframe = callback.timeframe
                        
                        if symbol not in symbol_timeframe_map:
                            symbol_timeframe_map[symbol] = {}
                        
                        if timeframe not in symbol_timeframe_map[symbol]:
                            symbol_timeframe_map[symbol][timeframe] = []
                            
                        symbol_timeframe_map[symbol][timeframe].append(callback)
                    
                    # Now process each symbol once
                    for symbol, timeframes in symbol_timeframe_map.items():
                        try:
                            # Get latest tick data - ONLY ONCE PER SYMBOL
                            if mt5:
                                tick = mt5.symbol_info_tick(symbol)
                                if tick:
                                    # Create an enhanced tick object with symbol information
                                    enhanced_tick_data = {'original_tick': tick, 'symbol': symbol}
                                    
                                    # Log tick data periodically (every 10th tick to avoid spam)
                                    if not hasattr(self, '_tick_counters'):
                                        self._tick_counters = {}
                                    if symbol not in self._tick_counters:
                                        self._tick_counters[symbol] = 0
                                    
                                    self._tick_counters[symbol] += 1
                                    if self._tick_counters[symbol] % 10 == 0:
                                        bid = getattr(tick, 'bid', 0)
                                        ask = getattr(tick, 'ask', 0)
                                        logger.debug(f"📊 {symbol} tick: Bid={bid:.5f} Ask={ask:.5f} Spread={(ask-bid)*10000:.1f} points")
                                    
                                    # Process tick for each timeframe's callbacks (just one tick per symbol)
                                    for tf, tf_callbacks in timeframes.items():
                                        for callback in tf_callbacks:
                                            # Limit to one tick processing per symbol/timeframe pair
                                            asyncio.create_task(callback.on_tick(enhanced_tick_data))
                                
                                # Get the latest candles for each timeframe
                                for timeframe, tf_callbacks in timeframes.items():
                                    mt5_timeframe = self._get_mt5_timeframe(timeframe)
                                    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 100)
                                    
                                    if rates is not None and len(rates) > 0:
                                        # Store last candle time to detect new candles
                                        candle_key = f"{symbol}_{timeframe}"
                                        if not hasattr(self, '_last_candle_times'):
                                            self._last_candle_times = {}
                                            
                                        # Get the time of the latest candle
                                        latest_time = pd.to_datetime(rates[-1]['time'], unit='s')
                                        
                                        # Check if this is a new candle
                                        is_new_candle = False
                                        if candle_key not in self._last_candle_times:
                                            is_new_candle = True
                                        elif latest_time != self._last_candle_times[candle_key]:
                                            is_new_candle = True
                                            
                                        if is_new_candle:
                                            # Get candle details for logging
                                            latest_candle = rates[-1]
                                            candle_open = latest_candle['open']
                                            candle_close = latest_candle['close']
                                            candle_high = latest_candle['high']
                                            candle_low = latest_candle['low']
                                            
                                            # Determine candle type and color
                                            candle_emoji = "🟢" if candle_close > candle_open else "🔴"
                                            candle_change = (candle_close - candle_open) / candle_open * 100 if candle_open > 0 else 0
                                            
                                            # Log the new candle formation
                                            logger.info(f"{candle_emoji} New {symbol}/{timeframe} candle: O:{candle_open:.5f} H:{candle_high:.5f} L:{candle_low:.5f} C:{candle_close:.5f} ({'+' if candle_change >= 0 else ''}{candle_change:.2f}%)")
                                            
                                            # Update the last candle time
                                            self._last_candle_times[candle_key] = latest_time
                                        
                                        # Distribute candle data to all callbacks for this symbol/timeframe
                                        for callback in tf_callbacks:
                                            asyncio.create_task(callback.on_new_candle(rates))
                                    else:
                                        logger.warning(f"⚠️ No candle data available for {symbol}/{timeframe}")
                            else:
                                logger.error("❌ MT5 not initialized in monitoring loop")
                                
                        except Exception as e:
                            logger.error(f"❌ Error processing {symbol}: {str(e)}")
                    
                    # Sleep to avoid excessive CPU usage
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(5)  # Sleep longer on error
                    
            logger.info("🛑 Monitoring loop ended")
            
        # Create and store the monitoring task
        self.rt_monitoring["task"] = asyncio.create_task(monitoring_loop())
        
    async def _reconnect(self):
        """Attempt to reconnect to MT5."""
        try:
            # Shutdown existing connection
            self.shutdown()
            
            # Initialize new connection
            success = self.initialize()
            
            if success:
                logger.info("Successfully reconnected to MT5")
                return True
            else:
                logger.error("Failed to reconnect to MT5")
                return False
                
        except Exception as e:
            logger.error(f"Error during MT5 reconnection: {str(e)}")
            return False

    def stop_real_time_monitoring(self):
        """Stop real-time market data monitoring."""
        logger.info("Stopping real-time market data monitoring")
        
        # Cancel monitoring task
        if self.rt_monitoring.get("task"):
            self.rt_monitoring["task"].cancel()
            self.rt_monitoring["task"] = None
            
        # Clear callbacks
        self.rt_monitoring["callbacks"] = {}
        
        # Set disabled flag
        self.rt_monitoring["running"] = False
        
        logger.info("Real-time monitoring stopped")
        return True
    
    def get_real_time_status(self):
        """Get the status of real-time monitoring."""
        # Ensure rt_monitoring is properly initialized
        if not hasattr(self, 'rt_monitoring') or not isinstance(self.rt_monitoring, dict):
            return {
                "running": False,
                "symbols": [],
                "callback_count": 0,
                "task_running": False
            }

        # Safe access to rt_monitoring
        callbacks = self.rt_monitoring.get("callbacks", [])
        symbols = set()
        
        # Extract symbols from callbacks
        for callback in callbacks:
            if hasattr(callback, 'symbol'):
                symbols.add(callback.symbol)
        
        # Check task status safely
        task = self.rt_monitoring.get("task")
        task_running = False
        if task is not None and hasattr(task, "done"):
            task_running = not task.done()
        
        return {
            "running": self.rt_monitoring.get("running", False),
            "symbols": list(symbols),
            "callback_count": len(callbacks),
            "task_running": task_running
        }

    async def on_tick(self, tick_data):
        """
        Process incoming tick data from MT5.
        
        Args:
            tick_data: Raw tick data from MT5 or an enhanced object with symbol information
            
        Returns:
            Processed tick data ready for analysis
        """
        try:
            if not tick_data:
                logger.warning("Received empty tick data")
                return None
            
            # Debug the type of tick_data to help troubleshoot
            logger.debug(f"Tick data type: {type(tick_data)}, data: {str(tick_data)[:200]}...")
                
            # Extract symbol from tick data
            symbol = None
            original_tick = None
            bid = 0
            ask = 0
            timestamp = datetime.now(UTC)
            volume = 0
            
            # Check if we have an enhanced tick object with symbol and original_tick
            if isinstance(tick_data, dict) and 'original_tick' in tick_data and 'symbol' in tick_data:
                symbol = tick_data['symbol']
                original_tick = tick_data['original_tick']
            else:
                # Legacy code path for direct tick objects
                original_tick = tick_data
                
                # If tick_data is a string, it might be the symbol itself
                if isinstance(tick_data, str):
                    symbol = tick_data
                
                # Safe attribute access
                elif hasattr(tick_data, 'symbol'):
                    symbol = tick_data.symbol
                
                # Try to extract from named tuple
                elif hasattr(tick_data, '_asdict') and callable(tick_data._asdict):
                    try:
                        tick_dict = tick_data._asdict()
                        if isinstance(tick_dict, dict) and 'symbol' in tick_dict:
                            symbol = tick_dict['symbol']
                    except Exception:
                        pass
                
                # Try to extract from dictionary
                elif isinstance(tick_data, dict):
                    symbol = tick_data.get('symbol')
                
                # Try to get the symbol from the callback metadata if available
                if not symbol and hasattr(self, 'current_symbol'):
                    symbol = self.current_symbol
            
            # Extract other fields from the original tick
            if original_tick is not None:
                # Safely extract other fields
                if hasattr(original_tick, 'bid'):
                    bid = original_tick.bid
                elif isinstance(original_tick, dict):
                    bid = original_tick.get('bid', 0)
                    
                if hasattr(original_tick, 'ask'):
                    ask = original_tick.ask
                elif isinstance(original_tick, dict):
                    ask = original_tick.get('ask', 0)
                    
                if hasattr(original_tick, 'time'):
                    try:
                        timestamp = datetime.fromtimestamp(original_tick.time)
                    except (ValueError, TypeError, OSError):
                        pass
                elif isinstance(original_tick, dict) and 'time' in original_tick:
                    try:
                        time_value = original_tick.get('time')
                        if time_value is not None:
                            timestamp = datetime.fromtimestamp(float(time_value))
                    except (ValueError, TypeError, OSError):
                        pass
                    
                if hasattr(original_tick, 'volume'):
                    volume = original_tick.volume
                elif isinstance(original_tick, dict):
                    volume = original_tick.get('volume', 0)
            
            # If still no symbol, log warning with more details and return None
            if not symbol:
                logger.warning(f"Tick data missing symbol information - cannot process. Data type: {type(tick_data)}")
                return None
                
            # Format tick data into a standardized dictionary
            processed_tick = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'time': timestamp,
                'volume': volume,
            }
            
            # Calculate spread from bid/ask
            processed_tick['spread'] = (ask - bid) if bid > 0 and ask > 0 else 0
            
            # Call the real-time data callbacks if registered
            self._notify_tick_callbacks(symbol, processed_tick)
            
            return processed_tick
            
        except Exception as e:
            logger.error(f"Error processing tick data: {str(e)}")
            logger.error(traceback.format_exc())  # Add full traceback for better debugging
            return None
            
    async def on_candle(self, candle_data, symbol=None, timeframe=None):
        """
        Process completed candle data from MT5.
        
        Args:
            candle_data: Raw candle data from MT5
            symbol: Optional symbol name if not contained in the candle data
            timeframe: Optional timeframe identifier
            
        Returns:
            Processed candle data ready for analysis
        """
        try:
            # Check if candle_data is empty - handle all possible types
            if candle_data is None:
                logger.warning("Received None candle_data")
                return None
            elif isinstance(candle_data, (list, tuple)) and len(candle_data) == 0:
                logger.warning("Received empty list/tuple candle_data")
                return None
            elif isinstance(candle_data, np.ndarray) and candle_data.size == 0:
                logger.warning("Received empty numpy array candle_data")
                return None
            elif isinstance(candle_data, pd.DataFrame) and candle_data.empty:
                logger.warning("Received empty DataFrame candle_data")
                return None
            
            # Format candle data into DataFrame
            try:
                df_candles = self.format_candles_to_dataframe(candle_data, symbol, timeframe)
            except Exception as e:
                logger.error(f"Error formatting candles to DataFrame: {str(e)}")
                logger.error(traceback.format_exc())
                return None
            
            if df_candles is None or df_candles.empty:
                logger.warning(f"Could not format candles for {symbol}/{timeframe}")
                return None
            
            # Process the candle data for analysis - with special error handling
            try:
                processed_candles = self.process_candles_data(df_candles)
            except Exception as e:
                logger.error(f"Error in process_candles_data: {str(e)}")
                logger.error(traceback.format_exc())
                # Return the original dataframe as a fallback
                processed_candles = df_candles
            
            # Generate higher timeframes if needed
            if timeframe in ['M1', '1m', 'm1', '1']:
                # Dictionary to hold multiple timeframes
                timeframe_data = {
                    timeframe: processed_candles
                }
                
                # Create higher timeframes
                for higher_tf in ['5m', '15m', '1h', '4h']:
                    try:
                        resampled_df = self.resample_to_higher_timeframe(processed_candles, higher_tf)
                        if resampled_df is not None and not resampled_df.empty:
                            timeframe_data[higher_tf] = resampled_df
                    except Exception as resample_err:
                        logger.error(f"Error resampling to {higher_tf}: {str(resample_err)}")
                
                # Notify callbacks with all timeframe data
                if symbol:
                    self._notify_candle_callbacks(symbol, timeframe, timeframe_data)
                
                return timeframe_data
            else:
                # If not base timeframe, just return the processed candles
                if symbol:
                    self._notify_candle_callbacks(symbol, timeframe, {timeframe: processed_candles})
                
                return {timeframe: processed_candles}
                
        except Exception as e:
            logger.error(f"Error processing candle data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def format_candles_to_dataframe(self, candles, symbol=None, timeframe=None):
        """
        Convert raw candle data to pandas DataFrame format.
        
        Args:
            candles: Raw candle data from MT5
            symbol: Optional symbol name
            timeframe: Optional timeframe identifier
            
        Returns:
            pd.DataFrame: Formatted candle data
        """
        try:
            # Check for empty candles - handle different types safely
            if candles is None:
                logger.warning(f"No candle data provided for {symbol}/{timeframe} (None)")
                return None
            elif isinstance(candles, (list, tuple)) and len(candles) == 0:
                logger.warning(f"No candle data provided for {symbol}/{timeframe} (empty list/tuple)")
                return None
            elif isinstance(candles, np.ndarray) and candles.size == 0:
                logger.warning(f"No candle data provided for {symbol}/{timeframe} (empty numpy array)")
                return None
            elif isinstance(candles, pd.DataFrame) and candles.empty:
                logger.warning(f"No candle data provided for {symbol}/{timeframe} (empty DataFrame)")
                return None
                
            # If candles is already a DataFrame
            if isinstance(candles, pd.DataFrame):
                df = candles.copy()
                
                # Add volume column if missing
                if 'volume' not in df.columns:
                    # Try to use tick_volume if available
                    if 'tick_volume' in df.columns:
                        df['volume'] = df['tick_volume']
                    else:
                        df['volume'] = 0
                
                # Add symbol column if not present and symbol is provided
                if 'symbol' not in df.columns and symbol:
                    df['symbol'] = symbol
                
                return df
            
            # Convert rates to pandas DataFrame
            df = pd.DataFrame(candles)
            
            # Rename columns if necessary
            if 'time' in df.columns:
                # Convert time to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Standard column set
            if all(col in df.columns for col in [0, 1, 2, 3, 4]):
                # Numeric columns format from MT5
                column_mapping = {0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close'}
                if 5 in df.columns:
                    column_mapping[5] = 'volume'
                df = df.rename(columns=column_mapping)
                
                # Add volume column if missing
                if 'volume' not in df.columns:
                    if 'real_volume' in df.columns:
                        df['volume'] = df['real_volume']
                    elif 'tick_volume' in df.columns:
                        df['volume'] = df['tick_volume']
                    else:
                        df['volume'] = 0
                    
                df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
                
                # Add symbol column if symbol is provided
                if symbol:
                    df['symbol'] = symbol
                
                # Add timeframe column if timeframe is provided
                if timeframe:
                    df['timeframe'] = timeframe
                
                # Sort by time index
                df = df.sort_index()
                
                return df
            else:
                logger.warning(f"No 'time' column found in candle data for {symbol}/{timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"Error formatting candles to DataFrame: {str(e)}")
            logger.error(traceback.format_exc())  # Add traceback for better debugging
            return None
            
    def process_candles_data(self, df):
        """
        Process candle data for further analysis.
        
        Args:
            df: DataFrame containing candle data
            
        Returns:
            pd.DataFrame: Processed candle data
        """
        try:
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for processing")
                return df
            
            # Make a copy to avoid modifying the original
            df_processed = df.copy()
            
            # Reset index if time is the index
            if df_processed.index.name == 'time':
                df_processed = df_processed.reset_index()
            
            # Calculate basic price changes
            if 'close' in df_processed.columns and 'open' in df_processed.columns:
                # Calculate candle body size
                df_processed['body_size'] = abs(df_processed['close'] - df_processed['open'])
                
                # Calculate total candle size (high to low)
                if 'high' in df_processed.columns and 'low' in df_processed.columns:
                    df_processed['candle_size'] = df_processed['high'] - df_processed['low']
                
                # Calculate candle direction (1=up, -1=down, 0=doji)
                # Use element-wise comparison with .gt(), .lt() and .eq() to avoid ambiguous truth value errors
                up_candles = df_processed['close'].gt(df_processed['open'])
                down_candles = df_processed['close'].lt(df_processed['open'])
                doji_candles = df_processed['close'].eq(df_processed['open'])
                
                df_processed['direction'] = np.where(up_candles, 1, 
                                                  np.where(down_candles, -1, 0))
                
                # Calculate percentage change - specify fill_method=None to prevent FutureWarning
                df_processed['pct_change'] = df_processed['close'].pct_change(fill_method=None) * 100
            
            # Ensure the DataFrame has a time column
            if 'time' not in df_processed.columns and df_processed.index.name != 'time':
                logger.warning("DataFrame lacks a time column")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error processing candle data: {str(e)}")
            logger.error(traceback.format_exc())  # Add full traceback for better debugging
            return df
            
    def resample_to_higher_timeframe(self, df, timeframe='5m'):
        """
        Resample data to a higher timeframe.
        
        Args:
            df: DataFrame containing candle data
            timeframe: Target timeframe for resampling
            
        Returns:
            pd.DataFrame: Resampled candle data
        """
        try:
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for resampling")
                return None
            
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure df has a datetime index
            if df_copy.index.name != 'time' and 'time' in df_copy.columns:
                df_copy = df_copy.set_index('time')
            
            # Map timeframe strings to pandas resampling rules
            tf_map = {
                '1m': '1min', 'm1': '1min', 'M1': '1min', '1': '1min',
                '5m': '5min', 'm5': '5min', 'M5': '5min', '5': '5min',
                '15m': '15min', 'm15': '15min', 'M15': '15min', '15': '15min',
                '30m': '30min', 'm30': '30min', 'M30': '30min', '30': '30min',
                '1h': '1h', 'h1': '1h', 'H1': '1h', '60': '1h',
                '4h': '4h', 'h4': '4h', 'H4': '4h', '240': '4h',
                '1d': '1D', 'd1': '1D', 'D1': '1D', '1440': '1D',
                '1w': '1W', 'w1': '1W', 'W1': '1W'
            }
            
            # Get the appropriate resampling rule
            rule = tf_map.get(timeframe, '5min')  # Default to 5min if timeframe not found
            
            # Ensure all required columns exist, adding defaults if needed
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_copy.columns:
                    # Add missing column with default value (0 for volume, first column value for others)
                    if col == 'volume':
                        df_copy[col] = 0
                    elif col == 'open' and 'close' in df_copy.columns:
                        df_copy[col] = df_copy['close']
                    elif col == 'high' and 'close' in df_copy.columns:
                        df_copy[col] = df_copy['close']
                    elif col == 'low' and 'close' in df_copy.columns:
                        df_copy[col] = df_copy['close']
                    elif col == 'close' and len(df_copy.columns) > 0:
                        # Use first numeric column as close if available
                        for potential_col in df_copy.columns:
                            if pd.api.types.is_numeric_dtype(df_copy[potential_col]):
                                df_copy[col] = df_copy[potential_col]
                                break
                        else:
                            df_copy[col] = 0
                    else:
                        df_copy[col] = 0
            
            # Resample OHLC data
            if all(col in df_copy.columns for col in required_columns):
                # Use try/except for the resampling operation
                try:
                    resampled = df_copy.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                    
                    # Preserve any other columns that need aggregation
                    for col in df_copy.columns:
                        if col not in required_columns and col != 'time':
                            if col in ['symbol', 'timeframe']:
                                # For categorical columns, use first value
                                resampled[col] = df_copy[col].resample(rule).first()
                    
                    # Reset index to have 'time' as a column
                    resampled = resampled.reset_index()
                    
                    # Update timeframe column
                    if 'timeframe' in resampled.columns:
                        resampled['timeframe'] = timeframe
                    
                    # Process the resampled data with error handling
                    try:
                        return self.process_candles_data(resampled)
                    except Exception as process_error:
                        logger.error(f"Error processing resampled data: {str(process_error)}")
                        logger.error(traceback.format_exc())
                        # Return the unprocessed resampled data as fallback
                        return resampled
                
                except Exception as resample_error:
                    logger.error(f"Error during resampling operation: {str(resample_error)}")
                    logger.error(traceback.format_exc())
                    return None
            else:
                missing_cols = [col for col in required_columns if col not in df_copy.columns]
                logger.warning(f"DataFrame missing required OHLC columns for resampling: {missing_cols}")
                return None
                
        except Exception as e:
            logger.error(f"Error resampling to higher timeframe: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def _notify_tick_callbacks(self, symbol, tick_data):
        """
        Notify registered callbacks about new tick data.
        
        Args:
            symbol: Symbol identifier
            tick_data: Processed tick data
        """
        if hasattr(self, 'real_time_callbacks') and self.real_time_callbacks:
            for callback in self.real_time_callbacks:
                if callback.symbol == symbol:
                    try:
                        # Call the on_tick method of the callback
                        callback.on_tick(tick_data)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {str(e)}")
    
    def _notify_candle_callbacks(self, symbol, timeframe, candle_data):
        """
        Notify registered callbacks about new candle data.
        
        Args:
            symbol: Symbol identifier
            timeframe: Timeframe identifier
            candle_data: Processed candle data dictionary
        """
        if hasattr(self, 'real_time_callbacks') and self.real_time_callbacks:
            for callback in self.real_time_callbacks:
                if callback.symbol == symbol and callback.timeframe == timeframe:
                    try:
                        # Call the on_new_candle method of the callback
                        callback.on_new_candle(candle_data)
                    except Exception as e:
                        logger.error(f"Error in candle callback: {str(e)}")

    async def get_rates(self, symbol: str, timeframe: str, count: int = 1000, start_pos: int = 0) -> Optional[pd.DataFrame]:
        """
        Get historical rate data from MT5 asynchronously.
        
        Args:
            symbol: Symbol to get rates for
            timeframe: Timeframe as string (e.g., 'M15', 'H1')
            count: Number of candles to retrieve
            start_pos: Start position (0 = current)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with rate data or None if error
        """
        try:
            if not self.connected:
                logger.error(f"MT5 not connected when getting rates for {symbol}/{timeframe}")
                return None
                
            # Map timeframe string to MT5 constant
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
            
            tf = timeframe_map.get(timeframe.upper())
            if tf is None:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
                
            # Select the symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
                
            # Run the actual data fetch in a separate thread to not block asyncio
            loop = asyncio.get_running_loop()
            rates = await loop.run_in_executor(
                None, 
                lambda: mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
            )
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.error(f"Failed to get rates for {symbol}/{timeframe}. Error: {error}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Ensure the DataFrame has all required columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    if col == 'volume' and 'tick_volume' in df.columns:
                        df['volume'] = df['tick_volume']
                    else:
                        logger.warning(f"Column {col} missing in rates data for {symbol}/{timeframe}")
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            logger.debug(f"Retrieved {len(df)} candles for {symbol}/{timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting rates for {symbol}/{timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def get_market_data_batch(self, symbols, timeframes, num_candles=1000):
        """
        Fetch data for multiple symbols and timeframes efficiently.
        
        Args:
            symbols: List of symbols to fetch data for
            timeframes: List of timeframes to fetch data for
            num_candles: Number of candles to fetch for each symbol/timeframe
            
        Returns:
            Nested dictionary with format {symbol: {timeframe: dataframe}}
        """
        results = {}
        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                try:
                    results[symbol][timeframe] = self.get_market_data(symbol, timeframe, num_candles)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
                    results[symbol][timeframe] = None
        return results
    
    def get_last_tick(self, symbol: str):
        """
        Get the latest price tick for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., 'EURUSD')
            
        Returns:
            The latest tick info or None if not available
        """
        try:
            if not self.connected:
                logger.error("MT5 not connected, cannot get last tick")
                return None
                
            # Ensure the symbol is selected
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
                
            # Get the last tick
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.warning(f"No tick data available for {symbol}")
                return None
                
            # Log tick details at debug level
            logger.debug(f"Latest tick for {symbol}: Bid={tick.bid}, Ask={tick.ask}, Time={tick.time}")
            
            return tick
            
        except Exception as e:
            logger.error(f"Error getting last tick for {symbol}: {str(e)}")
            return None
    
    def preprocess_data(self, data):
        """
        Apply basic preprocessing to raw MT5 data.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if data is None:
            return None
            
        # Handle missing values
        data = data.dropna()
        
        # Ensure datetime index is properly formatted
        if not data.empty and not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"Error converting index to datetime: {str(e)}")
        
        # Ensure column names are standardized
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        
        return data
    
    def get_monitored_symbols(self):
        """
        Get the list of currently monitored symbols.
        
        Returns:
            list: List of symbols currently being monitored
        """
        # Ensure rt_monitoring is properly initialized
        if not hasattr(self, 'rt_monitoring') or not isinstance(self.rt_monitoring, dict):
            return []

        # Extract symbols from callbacks
        symbols = set()
        for callback in self.rt_monitoring.get("callbacks", []):
            if hasattr(callback, 'symbol'):
                symbols.add(callback.symbol)
        
        return list(symbols)
    
    def subscribe_symbols(self, symbols, timeframes=None, callback_function=None):
        """
        Add new symbols to real-time monitoring.
        
        Args:
            symbols: List of symbols to add to monitoring
            timeframes: List of timeframes to monitor for these symbols
            callback_function: Function to call with processed data
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Adding {len(symbols)} symbols to real-time monitoring: {', '.join(symbols)}")
        
        # Initialize real-time monitoring if needed
        if not hasattr(self, 'rt_monitoring'):
            self.rt_monitoring = {
                "running": False,
                "task": None,
                "callbacks": []
            }
        
        # Get callback function from existing callbacks if not provided
        if not callback_function and self.rt_monitoring.get("callbacks"):
            for callback in self.rt_monitoring.get("callbacks", []):
                if hasattr(callback, 'callback_function'):
                    callback_function = callback.callback_function
                    break
        
        # If still no callback function, we can't proceed
        if not callback_function:
            logger.error("No callback function available for new symbols")
            return False
        
        # Get timeframes from existing callbacks if not provided
        if not timeframes and self.rt_monitoring.get("callbacks"):
            timeframes_set = set()
            for callback in self.rt_monitoring.get("callbacks", []):
                if hasattr(callback, 'timeframe'):
                    timeframes_set.add(callback.timeframe)
            
            if timeframes_set:
                timeframes = list(timeframes_set)
        
        # If still no timeframes, use defaults
        if not timeframes:
            timeframes = ["M15"]
        
        # Create callback objects for each symbol/timeframe combination
        for symbol in symbols:
            for timeframe in timeframes:
                # Check if callback already exists for this symbol/timeframe
                callback_exists = False
                for callback in self.rt_monitoring.get("callbacks", []):
                    if (hasattr(callback, 'symbol') and hasattr(callback, 'timeframe') and 
                        callback.symbol == symbol and callback.timeframe == timeframe):
                        callback_exists = True
                        break
                
                if not callback_exists:
                    callback = RealTimeDataCallback(symbol, timeframe, callback_function)
                    self.rt_monitoring["callbacks"].append(callback)
                    logger.info(f"Added callback for {symbol}/{timeframe}")
        
        # Start monitoring if not already running
        if not self.rt_monitoring.get("running", False):
            self.rt_monitoring["running"] = True
            self._start_monitoring_task()
            logger.info("Started real-time monitoring")
            
        return True
    
    def unsubscribe_symbols(self, symbols):
        """
        Remove symbols from real-time monitoring.
        
        Args:
            symbols: List of symbols to remove from monitoring
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Removing {len(symbols)} symbols from real-time monitoring: {', '.join(symbols)}")
        
        # Ensure rt_monitoring is properly initialized
        if not hasattr(self, 'rt_monitoring') or not isinstance(self.rt_monitoring, dict):
            logger.warning("Real-time monitoring not initialized, nothing to unsubscribe")
            return False
        
        if not self.rt_monitoring.get("callbacks"):
            logger.warning("No callbacks registered, nothing to unsubscribe")
            return False
        
        # Convert symbols to lowercase for case-insensitive comparison
        symbols_lower = [s.lower() for s in symbols]
        
        # Remove callbacks for the specified symbols
        new_callbacks = []
        for callback in self.rt_monitoring.get("callbacks", []):
            if hasattr(callback, 'symbol') and callback.symbol.lower() in symbols_lower:
                logger.debug(f"Removing callback for {callback.symbol}/{callback.timeframe}")
            else:
                new_callbacks.append(callback)
        
        # Update callbacks list
        self.rt_monitoring["callbacks"] = new_callbacks
        
        # Check if we still have any callbacks
        if not new_callbacks:
            logger.info("No callbacks remaining, stopping real-time monitoring")
            self.stop_real_time_monitoring()
        
        return True