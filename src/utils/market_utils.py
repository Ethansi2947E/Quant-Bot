from typing import Dict, Optional, Any, List, Tuple
from loguru import logger
import traceback
import pandas as pd
import numpy as np

def calculate_pip_value(symbol: str, symbol_info: Any = None, mt5_handler=None) -> float:
    """
    Calculate the pip value for a given symbol.
    
    Args:
        symbol: The trading symbol
        symbol_info: MT5 symbol_info object (optional)
        mt5_handler: MT5Handler instance (optional, used if symbol_info is not provided)
        
    Returns:
        float: The pip value for the symbol
    """
    try:
        # Default pip value (fallback)
        pip_value = 0.0001  # Default for forex 4-digit pairs
        
        # If symbol_info not provided and mt5_handler is available, get symbol info
        if symbol_info is None and mt5_handler is not None:
            symbol_info = mt5_handler.get_symbol_info(symbol)
        
        if symbol_info:
            # Get point value and digits from symbol info
            point = symbol_info.point
            digits = symbol_info.digits
            
            # Calculate pip value based on digits (standard practice in forex and CFDs)
            if digits == 3 or digits == 5:  # 3 or 5 decimal places (common for forex)
                pip_value = point * 10  # 1 pip = 10 points
            elif digits == 2:  # 2 decimal places (common for JPY pairs)
                pip_value = point * 10  # 1 pip = 10 points
            elif digits == 1:  # 1 decimal place (some commodities and indices)
                pip_value = point * 10  # 1 pip = 10 points
            elif digits == 0:  # 0 decimal places (some cryptocurrencies)
                pip_value = point * 10  # 1 pip = 10 points
            else:
                pip_value = point  # Use point as pip for other cases
                
            logger.debug(f"Calculated pip value for {symbol}: point={point}, digits={digits}, pip_value={pip_value}")
            return pip_value
        
        # Fallback to hard-coded values if MT5 info not available
        # This should rarely happen but provides a safety mechanism
        if symbol.startswith("GOLD") or symbol.startswith("XAU"):
            pip_value = 0.01  # For gold
        elif symbol.startswith("US") or symbol.startswith("NDX") or symbol.startswith("SPX"):
            pip_value = 0.01  # For US indices
        elif any(symbol.startswith(prefix) for prefix in ["Crash", "Boom", "Jump", "Volatility", "Range", "Step"]):
            pip_value = 0.001  # Synthetic indices typically have 3 digits
        elif symbol.endswith("JPY"):
            pip_value = 0.01  # For JPY pairs
        
        logger.warning(f"Using fallback pip value for {symbol}: {pip_value}")
        return pip_value
        
    except Exception as e:
        logger.error(f"Error calculating pip value for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return 0.0001  # Return default value in case of error
    
def convert_pips_to_price(pips: float, symbol: str, symbol_info: Any = None, mt5_handler=None) -> float:
    """
    Convert pips to price value for a symbol.
    
    Args:
        pips: Number of pips
        symbol: The trading symbol
        symbol_info: MT5 symbol_info object (optional)
        mt5_handler: MT5Handler instance (optional, used if symbol_info is not provided)
        
    Returns:
        float: The price equivalent of the given pips
    """
    pip_value = calculate_pip_value(symbol, symbol_info, mt5_handler)
    return pips * pip_value

def convert_price_to_pips(price_diff: float, symbol: str, symbol_info: Any = None, mt5_handler=None) -> float:
    """
    Convert price difference to pips for a symbol.
    
    Args:
        price_diff: Price difference
        symbol: The trading symbol
        symbol_info: MT5 symbol_info object (optional)
        mt5_handler: MT5Handler instance (optional, used if symbol_info is not provided)
        
    Returns:
        float: The pip equivalent of the given price difference
    """
    pip_value = calculate_pip_value(symbol, symbol_info, mt5_handler)
    if pip_value == 0:
        return 0  # Avoid division by zero
    return price_diff / pip_value 