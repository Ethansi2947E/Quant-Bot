from typing import Dict, Optional, List, Tuple
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta
import json

from config.config import TRADING_CONFIG
from src.models import Trade

class RiskManager:
    def __init__(self):
        self.risk_per_trade = TRADING_CONFIG["risk_per_trade"]
        self.max_daily_risk = TRADING_CONFIG["max_daily_risk"]
        self.open_trades: List[Dict] = []
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        symbol: str
    ) -> float:
        """Calculate position size based on risk parameters."""
        try:
            # Calculate risk amount in account currency
            risk_amount = account_balance * self.risk_per_trade
            
            # Calculate price difference for stop loss
            price_difference = abs(entry_price - stop_loss)
            
            if price_difference == 0:
                logger.error("Invalid stop loss - same as entry price")
                return 0.0
            
            # Calculate position size in standard lots
            position_size = risk_amount / (price_difference * 100000)
            
            # Round to 2 decimal places and ensure minimum size
            position_size = round(max(0.01, position_size), 2)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.01
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        atr_multiplier: float = 2.0
    ) -> float:
        """Calculate stop loss based on ATR."""
        try:
            # Get latest ATR value
            atr = df['atr'].iloc[-1]
            
            if signal_type == "BUY":
                stop_loss = entry_price - (atr * atr_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * atr_multiplier)
            
            return round(stop_loss, 5)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            # Default to 1% stop loss if ATR calculation fails
            return entry_price * (0.99 if signal_type == "BUY" else 1.01)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit based on risk-reward ratio."""
        try:
            # Calculate price difference for stop loss
            stop_loss_diff = abs(entry_price - stop_loss)
            
            # Calculate take profit distance
            take_profit_diff = stop_loss_diff * risk_reward_ratio
            
            # Calculate take profit price
            if entry_price > stop_loss:  # Long position
                take_profit = entry_price + take_profit_diff
            else:  # Short position
                take_profit = entry_price - take_profit_diff
            
            return round(take_profit, 5)
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            # Default to 2% take profit if calculation fails
            return entry_price * (1.02 if entry_price > stop_loss else 0.98)
    
    def check_daily_risk(self, account_balance: float) -> bool:
        """Check if daily risk limit has been reached."""
        try:
            # Calculate total risk for open positions
            total_risk = sum(
                trade.get("risk_amount", 0.0)
                for trade in self.open_trades
                if trade.get("entry_time", datetime.now()).date() == datetime.now().date()
            )
            
            # Calculate maximum allowed risk
            max_risk = account_balance * self.max_daily_risk
            
            return total_risk < max_risk
            
        except Exception as e:
            logger.error(f"Error checking daily risk: {str(e)}")
            return False
    
    def validate_trade(
        self,
        signal_type: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float
    ) -> Tuple[bool, str]:
        """Validate trade parameters."""
        try:
            # Check confidence threshold
            if confidence < 0.5:
                return False, "Low confidence signal"
            
            # Check price levels
            if signal_type == "BUY":
                if not (entry_price > stop_loss and take_profit > entry_price):
                    return False, "Invalid price levels for BUY order"
            else:  # SELL
                if not (entry_price < stop_loss and take_profit < entry_price):
                    return False, "Invalid price levels for SELL order"
            
            # Check minimum distance between prices (0.1%)
            min_distance = entry_price * 0.001
            if abs(entry_price - stop_loss) < min_distance:
                return False, "Stop loss too close to entry"
            if abs(entry_price - take_profit) < min_distance:
                return False, "Take profit too close to entry"
            
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def adjust_for_news(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        news_volatility: float = 1.0
    ) -> Tuple[float, float, float]:
        """Adjust trade parameters based on news volatility."""
        try:
            # Increase distances based on news volatility
            if news_volatility > 1.0:
                # Adjust stop loss
                sl_distance = abs(entry_price - stop_loss)
                new_sl_distance = sl_distance * news_volatility
                
                # Adjust take profit
                tp_distance = abs(entry_price - take_profit)
                new_tp_distance = tp_distance * news_volatility
                
                # Apply new levels maintaining direction
                if entry_price > stop_loss:  # Long position
                    stop_loss = entry_price - new_sl_distance
                    take_profit = entry_price + new_tp_distance
                else:  # Short position
                    stop_loss = entry_price + new_sl_distance
                    take_profit = entry_price - new_tp_distance
            
            return round(entry_price, 5), round(stop_loss, 5), round(take_profit, 5)
            
        except Exception as e:
            logger.error(f"Error adjusting for news: {str(e)}")
            return entry_price, stop_loss, take_profit
    
    def update_open_trades(self, trades: List[Dict]):
        """Update the list of open trades."""
        self.open_trades = trades
    
    def should_close_trade(
        self,
        trade: Dict,
        current_price: float,
        indicators: Dict
    ) -> Tuple[bool, str]:
        """Determine if a trade should be closed based on current conditions."""
        try:
            # Check if price hit stop loss or take profit
            if trade["direction"] == "LONG":
                if current_price <= trade["stop_loss"]:
                    return True, "Stop loss hit"
                if current_price >= trade["take_profit"]:
                    return True, "Take profit hit"
            else:  # SHORT
                if current_price >= trade["stop_loss"]:
                    return True, "Stop loss hit"
                if current_price <= trade["take_profit"]:
                    return True, "Take profit hit"
            
            # Check for trend reversal
            if trade["direction"] == "LONG" and indicators.get("structure") == "STRONG_DOWNTREND":
                return True, "Strong trend reversal"
            if trade["direction"] == "SHORT" and indicators.get("structure") == "STRONG_UPTREND":
                return True, "Strong trend reversal"
            
            return False, "Trade conditions valid"
            
        except Exception as e:
            logger.error(f"Error checking trade closure: {str(e)}")
            return False, f"Error: {str(e)}" 