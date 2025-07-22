import sys
import os
from typing import List, Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mt5_handler import MT5Handler
from loguru import logger

class MT5Service:
    """
    A service class to interact with the MT5Handler for live data.
    This acts as a bridge between the API and the core MT5 connection logic.
    """
    _instance: Optional['MT5Service'] = None
    _mt5_handler: Optional[MT5Handler] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MT5Service, cls).__new__(cls)
            cls._instance._initialize_handler()
        return cls._instance

    def _initialize_handler(self):
        """Initializes the MT5Handler instance."""
        if MT5Service._mt5_handler is None:
            logger.info("Initializing MT5Service and creating new MT5Handler instance...")
            try:
                # Use the singleton pattern from MT5Handler to get an instance
                MT5Service._mt5_handler = MT5Handler.get_instance()
                if not MT5Service._mt5_handler.connected:
                    logger.error("MT5Service failed to establish a connection via MT5Handler.")
                else:
                    logger.success("MT5Service successfully connected to MT5.")
            except Exception as e:
                logger.error(f"An exception occurred during MT5Handler initialization: {e}")
    
    def get_active_trades(self, magic_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetches currently open trades from the MT5 terminal.
        
        Args:
            magic_number: Optional magic number to filter trades.
        
        Returns:
            A list of dictionaries, where each dictionary represents an active trade.
        """
        if not self._mt5_handler or not self._mt5_handler.connected:
            logger.warning("MT5 handler not connected. Attempting to re-initialize...")
            self._initialize_handler()
            if not self._mt5_handler or not self._mt5_handler.connected:
                logger.error("Failed to get active trades: MT5 connection is not available.")
                return []
        
        try:
            # The 'type' field from MT5 is 0 for buy, 1 for sell. We need to convert this.
            positions = self._mt5_handler.get_open_positions(magic_number=magic_number)
            
            # Format the positions to match frontend expectations
            formatted_trades = []
            for pos in positions:
                formatted_trades.append({
                    "ticket": pos.get("ticket"),
                    "symbol": pos.get("symbol"),
                    "order_type": "buy" if pos.get("type") == 0 else "sell",
                    "open_time": pos.get("time"), # This is a timestamp
                    "open_price": pos.get("open_price"),
                    "current_price": pos.get("current_price"),
                    "volume": pos.get("volume"),
                    "sl": pos.get("sl"),
                    "tp": pos.get("tp"),
                    "profit": pos.get("profit"),
                    "swap": pos.get("swap"),
                    "comment": pos.get("comment"),
                    "magic": pos.get("magic"),
                })
            return formatted_trades
        except Exception as e:
            logger.error(f"An error occurred while fetching active trades: {e}")
            return []

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetches live account information (balance, equity, etc.) from the MT5 terminal.
        """
        if not self._mt5_handler or not self._mt5_handler.connected:
            logger.warning("MT5 handler not connected. Attempting to re-initialize...")
            self._initialize_handler()
            if not self._mt5_handler or not self._mt5_handler.connected:
                logger.error("Failed to get account info: MT5 connection is not available.")
                return None
        
        try:
            account_info = self._mt5_handler.get_account_info()
            return account_info
        except Exception as e:
            logger.error(f"An error occurred while fetching account info: {e}")
            return None

# Create a singleton instance of the service for the API to use
mt5_service = MT5Service() 