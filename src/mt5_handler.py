import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any, Tuple
import json

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
        """Place a market order."""
        if not self.connected:
            logger.error("MT5 not connected")
            return None
        
        action_map = {
            "BUY": mt5.ORDER_TYPE_BUY,
            "SELL": mt5.ORDER_TYPE_SELL
        }
        
        action = action_map.get(order_type)
        if action is None:
            logger.error(f"Invalid order type: {order_type}")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        point = symbol_info.point
        price = symbol_info.ask if action == mt5.ORDER_TYPE_BUY else symbol_info.bid
        
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
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order failed. Error: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed. Retcode: {result.retcode}")
            return None
        
        return {
            "ticket": result.order,
            "volume": result.volume,
            "price": result.price,
            "comment": comment
        }
    
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
            "volume": pos.volume,
            "open_price": pos.price_open,
            "current_price": pos.price_current,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit,
            "comment": pos.comment
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
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position. Error: {mt5.last_error()}")
            return False
        
        return True
    
    def __del__(self):
        """Cleanup MT5 connection."""
        if self.connected:
            mt5.shutdown()
            logger.info("MT5 connection closed") 