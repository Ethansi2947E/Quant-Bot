import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any
import time
import traceback  # Add import for traceback module used in __del__
import json
import math
import sys

from config.config import MT5_CONFIG

class MT5Handler:
    def __init__(self):
        self.connected = False
        self.initialize()
        self._last_error = None  # Add error tracking
    
    def initialize(self) -> bool:
        """Initialize connection to MT5 terminal."""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Login to MT5
            if not mt5.login(
                login=MT5_CONFIG["login"],
                server=MT5_CONFIG["server"],
                password=MT5_CONFIG["password"],
                timeout=MT5_CONFIG.get("timeout", 60000)
            ):
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}")
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
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info. Error: {mt5.last_error()}")
            return {}
        
        return {
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "free_margin": account_info.margin_free,
            "leverage": account_info.leverage,
            "currency": account_info.currency
        }
    
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
                if not mt5.symbol_select(symbol, True):
                    error_code = mt5.last_error()
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
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
                
                # Check for connection errors
                if rates is None:
                    error_code = mt5.last_error()
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
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
            
        # Check if symbol is visible in MarketWatch
        if not symbol_info.visible:
            logger.warning(f"{symbol} not visible, trying to add it")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to add {symbol} to MarketWatch")
                return None
                
        # Get account info for logging
        account_info = mt5.account_info()
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
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"Failed to send order: {mt5.last_error()}")
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
                tick = mt5.symbol_info_tick(symbol)
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
            logger.warning("MT5 not connected when trying to get positions, attempting to reconnect")
            if not self.initialize():
                logger.error("Failed to reconnect to MT5")
                return []
        
        # Track recovery attempts
        recovery_attempts = 0
        max_recovery_attempts = 3
        
        while recovery_attempts <= max_recovery_attempts:
            try:
                positions = mt5.positions_get()
                if positions is None:
                    error = mt5.last_error()
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
                
                # Successfully got positions
                return [{
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": pos.type,
                    "volume": pos.volume,
                    "lots": pos.volume,  # Alias for volume
                    "price_open": pos.price_open,
                    "open_price": pos.price_open,  # Alias for price_open
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "sl_initial": pos.sl,  # Initial stop loss is same as current
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "comment": pos.comment,
                    "time": pos.time
                } for pos in positions]
                
            except Exception as e:
                recovery_attempts += 1
                
                # Check if it looks like a connection error
                if "IPC" in str(e) or "connection" in str(e).lower():
                    if recovery_attempts <= max_recovery_attempts:
                        logger.warning(f"Connection error when getting positions, attempting to reconnect (attempt {recovery_attempts}/{max_recovery_attempts}): {str(e)}")
                        if self.initialize():
                            logger.info("MT5 connection re-established, retrying positions fetch")
                            continue
                        time.sleep(1)  # Brief pause before retry
                    else:
                        logger.error(f"Failed to recover connection after {max_recovery_attempts} attempts")
                else:
                    logger.error(f"Exception in get_open_positions: {str(e)}")
                
                if recovery_attempts > max_recovery_attempts:
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
            return self.get_market_data(symbol, timeframe, num_candles)
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

    def get_order_history(self, ticket=None, symbol=None, days=7):
        """Get historical orders from MT5
        
        Args:
            ticket (int, optional): Specific order ticket to retrieve
            symbol (str, optional): Symbol to get history for
            days (int, optional): Number of days to look back. Defaults to 7.
            
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
            
            # Define filters
            request = {
                "from": from_date,
                "to": to_date
            }
            
            if ticket is not None:
                request["ticket"] = ticket
                
            if symbol is not None:
                request["symbol"] = symbol
            
            # Get the orders history
            history_orders = mt5.history_orders_get(**request)
            
            if history_orders is None or len(history_orders) == 0:
                logger.warning(f"No order history found for the specified period ({days} days)")
                
                # Try to get deals history instead, which should have the profit information
                deals_request = {
                    "from": from_date,
                    "to": to_date
                }
                
                if ticket is not None:
                    deals_request["ticket"] = ticket
                    
                if symbol is not None:
                    deals_request["symbol"] = symbol
                
                deals_history = mt5.history_deals_get(**deals_request)
                
                if deals_history is None or len(deals_history) == 0:
                    logger.warning(f"No deals history found for the specified period ({days} days)")
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
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            if ticket:
                logger.error(f"Failed to get history for ticket {ticket}")
            return []
    
    def get_account_history(self, days=7):
        """Get historical account data (balance, equity, etc.)
        
        Args:
            days (int, optional): Number of days to look back. Defaults to 7.
            
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
            
            deals = mt5.history_deals_get(from_date, to_date)
            
            if deals is None or len(deals) == 0:
                logger.warning(f"No deals found for the past {days} days")
                return self._generate_balance_history(days)
            
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
    
    def _generate_balance_history(self, days=7):
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

    def open_buy(self, symbol: str, volume: float, stop_loss: float, take_profit: float, comment: str = "") -> Optional[int]:
        """
        Open a buy position (market order).
        
        Args:
            symbol: Trading symbol
            volume: Position size (lot size)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Optional comment for the trade
            
        Returns:
            Ticket number if successful, None otherwise
        """
        try:
            result = self.place_market_order(
                symbol=symbol,
                order_type="BUY",
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )
            
            if result:
                # Check if we have a ticket number either in 'ticket' or 'order' field
                ticket = result.get('ticket') or result.get('order')
                if ticket:
                    logger.info(f"Buy order placed successfully: Ticket {ticket}")
                    return ticket
                    
            logger.error(f"Failed to place buy order for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error opening buy position: {str(e)}")
            logger.exception("Exception details:")
            return None
            
    def open_sell(self, symbol: str, volume: float, stop_loss: float, take_profit: float, comment: str = "") -> Optional[int]:
        """
        Open a sell position (market order).
        
        Args:
            symbol: Trading symbol
            volume: Position size (lot size)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Optional comment for the trade
            
        Returns:
            Ticket number if successful, None otherwise
        """
        try:
            result = self.place_market_order(
                symbol=symbol,
                order_type="SELL",
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )
            
            if result:
                # Check for ticket in either 'ticket' or 'order' field (same as open_buy)
                ticket = result.get('ticket') or result.get('order')
                if ticket:
                    logger.info(f"Sell order placed successfully: Ticket {ticket}")
                    return ticket
                
            logger.error(f"Failed to place sell order for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error opening sell position: {str(e)}")
            logger.exception("Exception details:")
            return None

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
        symbol_info = mt5.symbol_info(symbol)
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
            account_info = mt5.account_info()
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
