import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any, Tuple
import json
import time

from config.config import MT5_CONFIG, TRADING_CONFIG

class MT5Handler:
    def __init__(self):
        self.connected = False
        self.initialize()
    
    def initialize(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if not mt5.initialize(
            login=MT5_CONFIG["login"],
            server=MT5_CONFIG["server"],
            password=MT5_CONFIG["password"],
            timeout=MT5_CONFIG["timeout"]
        ):
            logger.error(f"MT5 initialization failed. Error: {mt5.last_error()}")
            return False
        
        self.connected = True
        logger.info("MT5 connection established successfully")
        return True
    
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
        """Fetch market data from MT5."""
        if not self.connected:
            logger.error("MT5 not connected")
            return None
        
        # Convert timeframe string to MT5 timeframe constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        tf = timeframe_map.get(timeframe)
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_candles)
        if rates is None:
            logger.error(f"Failed to get market data. Error: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Check if required volume data exists
        if 'tick_volume' not in df.columns:
            logger.error(f"No tick volume data available for {symbol}")
            return None
            
        # Set volume to tick_volume for consistency
        df['volume'] = df['tick_volume']
        
        return df
    
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
        
        # Validate order type
        action_map = {
            "BUY": mt5.ORDER_TYPE_BUY,
            "SELL": mt5.ORDER_TYPE_SELL
        }
        action = action_map.get(order_type)
        if action is None:
            logger.error(f"Invalid order type: {order_type}")
            return None
        
        # Get symbol info and validate
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        # Ensure the symbol is selected for trading
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol {symbol} for trading; continuing anyway.")

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}, retrying after a short delay")
            time.sleep(0.5)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get tick data for {symbol} after retry")
                return None
            
        # Use proper price based on order type
        price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Validate stop loss and take profit levels
        if action == mt5.ORDER_TYPE_BUY:
            if stop_loss >= price:
                logger.error(f"Invalid stop loss for BUY order: SL ({stop_loss}) must be below entry ({price})")
                return None
            if take_profit <= price:
                logger.error(f"Invalid take profit for BUY order: TP ({take_profit}) must be above entry ({price})")
                return None
        else:  # SELL
            if stop_loss <= price:
                logger.error(f"Invalid stop loss for SELL order: SL ({stop_loss}) must be above entry ({price})")
                return None
            if take_profit >= price:
                logger.error(f"Invalid take profit for SELL order: TP ({take_profit}) must be below entry ({price})")
                return None
        
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
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Try to send order with retries
        max_retries = 3
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"Order failed. Error: {mt5.last_error()}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                    logger.warning(f"Requote detected on attempt {attempt + 1}")
                    request["deviation"] += 10
                    continue
                    
                error_msg = f"Order failed. Retcode: {result.retcode}"
                if hasattr(result, 'comment'):
                    error_msg += f", Comment: {result.comment}"
                logger.error(error_msg)
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                return None
            
            # Order successful
            return {
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": comment
            }
        
        return None
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        if not self.connected:
            logger.error("MT5 not connected")
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to get positions. Error: {mt5.last_error()}")
            return []
        
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
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 connection closed")
    
    def __del__(self):
        """Cleanup MT5 connection."""
        if self.connected:
            mt5.shutdown()
            logger.info("MT5 connection closed")

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