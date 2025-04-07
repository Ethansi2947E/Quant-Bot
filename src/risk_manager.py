import os
import json
import time
from datetime import datetime, timedelta, UTC
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from config.config import TRADING_CONFIG, TELEGRAM_CONFIG, RISK_MANAGER_CONFIG
from loguru import logger
import pandas as pd
import MetaTrader5 as mt5
import numpy as np

from src.mt5_handler import MT5Handler

class RiskManager:
    """Risk manager handles position sizing, risk control, and trade management."""

    def __init__(self, mt5_handler: Optional[MT5Handler] = None, timeframe: str = "M15"):
        """
        Initialize the risk manager with a MT5 handler and configuration.
        
        Args:
            mt5_handler: MetaTrader5 interface instance (optional)
            timeframe: The timeframe to use for risk calculations, defaults to "M15"
        """
        # Initialize MT5 handler
        self.mt5_handler = mt5_handler if mt5_handler is not None else MT5Handler()
        self.mt5 = mt5_handler  # Alias for compatibility
        
        # Store timeframe
        self.timeframe = timeframe
        
        # Get configuration from Risk Manager Config
        from config.config import get_risk_config
        self.config = get_risk_config(timeframe)
        
        # Set risk parameters from config
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01)
        self.max_daily_risk = self.config.get('max_daily_risk', 0.03)
        self.max_drawdown_pause = self.config.get('max_drawdown_pause', 0.05)
        self.max_daily_trades = self.config.get('max_daily_trades', 8)
        self.min_trades_spacing = self.config.get('min_trades_spacing', 1)
        self.max_concurrent_trades = self.config.get('max_concurrent_trades', 2)
        
        # Core risk parameters - use timeframe-specific values if available
        self.max_daily_loss = self.config.get('max_daily_loss', 0.02)
        self.max_weekly_loss = self.config.get('max_weekly_loss', 0.05)
        self.max_monthly_loss = self.config.get('max_monthly_loss', 0.10)
        self.max_drawdown = 0.05  # Default to 5% max drawdown
        
        # Position management - use timeframe-specific values if available
        self.max_weekly_trades = self.config.get('max_weekly_trades', 16)
        self.use_fixed_lot_size = TRADING_CONFIG['use_fixed_lot_size']  # Use global trading config
        self.fixed_lot_size = TRADING_CONFIG['fixed_lot_size']
        self.max_lot_size = TRADING_CONFIG['max_lot_size']
        
        # Drawdown controls
        self.consecutive_loss_limit = self.config.get('consecutive_loss_limit', 3)
        self.drawdown_position_scale = self.config.get('drawdown_position_scale', {
            0.02: 0.75,   # 75% size at 2% drawdown
            0.03: 0.50,   # 50% size at 3% drawdown
            0.04: 0.25,   # 25% size at 4% drawdown
            0.05: 0.0     # Stop trading at 5% drawdown
        })
        
        # Partial profit targets
        self.partial_tp_levels = self.config.get('partial_tp_levels', [
            {'size': 0.4, 'ratio': 0.5},  # 40% at 0.5R
            {'size': 0.3, 'ratio': 1.0},  # 30% at 1R
            {'size': 0.3, 'ratio': 1.5}   # 30% at 1.5R
        ])
        
        # Volatility-based position sizing
        self.volatility_position_scale = self.config.get('volatility_position_scale', {
            'extreme': 0.25,  # 25% size in extreme volatility
            'high': 0.50,     # 50% size in high volatility
            'normal': 1.0,    # Normal size
            'low': 0.75       # 75% size in low volatility
        })
        
        # Recovery mode
        self.recovery_mode = self.config.get('recovery_mode', {
            'enabled': True,
            'threshold': 0.05,        # 5% drawdown activates recovery
            'position_scale': 0.5,    # 50% position size
            'win_streak_required': 3,  # Need 3 winners to exit
            'max_trades_per_day': 2,   # Limited trades in recovery
            'min_win_rate': 0.40      # Min win rate to exit recovery
        })
        
        # Correlation controls
        self.correlation_limits = {
            'max_correlation': self.config.get('correlation_threshold', 0.8),
            'lookback_period': self.config.get('correlation_limits', {}).get('lookback_period', 20),
            'min_trades_for_calc': self.config.get('correlation_limits', {}).get('min_trades_for_calc', 50),
            'high_correlation_scale': self.config.get('correlation_limits', {}).get('high_correlation_scale', 0.5)
        }
        
        # Session-based risk adjustments
        self.session_risk_multipliers = self.config.get('session_risk_multipliers', {
            'london_open': 1.0,
            'london_ny_overlap': 1.0,
            'ny_open': 1.0,
            'asian': 0.5,
            'pre_news': 0.0,
            'post_news': 0.5
        })
        
        # Log the timeframe-specific parameters
        logger.info(f"RiskManager initialized with timeframe {timeframe} parameters")
        logger.debug(f"Timeframe-specific risk parameters: max_risk_per_trade={self.max_risk_per_trade}, "
                    f"max_daily_trades={self.max_daily_trades}, max_concurrent_trades={self.max_concurrent_trades}")
        
        # Add missing attributes identified by type checker
        self.market_condition_adjustments = {
            'trending': 1.2,
            'ranging': 1.0,
            'choppy': 0.5,
            'breakout': 1.1,
            'reversal': 0.8,
            'high_volatility': 0.75,
            'low_volatility': 0.9,
            'normal': 1.0
        }
        
        self.confidence_position_scale = {
            0.9: 1.0,  # Full size for high confidence (90%+)
            0.7: 0.8,  # 80% size for good confidence (70-89%)
            0.5: 0.5,  # 50% size for moderate confidence (50-69%)
            0.3: 0.3   # 30% size for low confidence (<50%)
        }
        
        self.dynamic_tp_levels = {
            'trending': [
                {'ratio': 1.0, 'size': 0.3},
                {'ratio': 2.0, 'size': 0.3},
                {'ratio': 3.0, 'size': 0.4}
            ],
            'ranging': [
                {'ratio': 0.5, 'size': 0.4},
                {'ratio': 1.0, 'size': 0.3},
                {'ratio': 1.5, 'size': 0.3}
            ],
            'breakout': [
                {'ratio': 1.5, 'size': 0.3},
                {'ratio': 2.5, 'size': 0.3},
                {'ratio': 3.5, 'size': 0.4}
            ],
            'reversal': [
                {'ratio': 0.75, 'size': 0.5},
                {'ratio': 1.5, 'size': 0.5}
            ]
        }
        
        # Track daily performance
        self.daily_stats = {
            'total_risk': 0.0,
            'realized_pnl': 0.0,
            'trade_count': 0,
            'starting_balance': 0.0,
            'last_reset': datetime.now(UTC).date()
        }

        self.open_trades: List[Dict[str, Any]] = []
    
        # Initialize starting balance
        self._update_starting_balance()

    
    def _update_starting_balance(self) -> None:
        """Update the starting balance from MT5 account info."""
        try:
            account_info = self._get_account_info()
            if account_info and 'balance' in account_info:
                self.daily_stats['starting_balance'] = account_info['balance']
        except Exception as e:
            logger.error(f"Error updating starting balance: {str(e)}")
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get account information from MT5."""
        try:
            if self.mt5_handler:
                return self.mt5_handler.get_account_info()
            
            # Fallback to direct MT5 call if no handler
            # Using type ignore since pyright doesn't recognize account_info as a method
            account_info = mt5.account_info()  # type: ignore
            if account_info is None:
                logger.error("Failed to get account info")
                return {}
            
            return {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
            
    def set_mt5_handler(self, mt5_handler: MT5Handler) -> None:
        """
        Set the MT5Handler instance for this RiskManager.
        
        Args:
            mt5_handler: The MT5Handler instance to use
        """
        logger.info("Setting MT5Handler in RiskManager")
        self.mt5_handler = mt5_handler

    def check_daily_limits(self, account_balance: float,
                           new_trade_risk: float) -> tuple[bool, str]:
        """
        Check if adding a new trade's risk will exceed the daily risk limit.
        Daily risk limit is defined as a percentage of the account balance.

        Args:
            account_balance (float): The current account balance.
            new_trade_risk (float): The risk amount for the new trade.

        Returns:
            tuple: (True, '') if trade is allowed, or (False, reason) if not.
        """
        try:
            # Ensure daily stats are up-to-date
            current_date = datetime.now(UTC).date()
            if self.daily_stats['last_reset'] < current_date:
                self.reset_daily_stats()
            
            # Calculate total risk including open positions
            current_risk = sum(
                trade.get("risk_amount", 0.0)
                for trade in self.open_trades
                if trade.get("entry_time", datetime.now(UTC)).date() == current_date
            )
            
            # Add new trade risk
            total_risk = current_risk + new_trade_risk
            allowed_risk = account_balance * self.max_daily_risk
            
            if total_risk > allowed_risk:
                return False, f"Daily risk limit of {allowed_risk:.2f} would be exceeded (Current: {current_risk:.2f}, New: {new_trade_risk:.2f})"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error checking daily limits: {str(e)}")
            return False, str(e)

    def _validate_position_inputs(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float
    ) -> bool:
        """
        Validate position sizing inputs to prevent errors.
        
        Args:
            account_balance: Account balance
            risk_per_trade: Risk per trade as decimal (e.g., 0.01 for 1%)
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        # Check if account balance is valid
        if account_balance <= 0.0:
            logger.error(f"Invalid account balance: {account_balance}")
            # Recover with a default balance rather than failing
            logger.warning("Using default account balance of 10000 for calculations")
            # Use a sensible default (won't modify the input parameter, but allows processing to continue)
            account_balance = 10000.0
            return True
            
        # Check if risk parameter is valid
        if risk_per_trade <= 0.0 or risk_per_trade > 1.0:
            logger.error(f"Invalid risk percentage: {risk_per_trade}")
            return False
            
        # Check if prices are valid
        if entry_price <= 0.0:
            logger.error(f"Invalid entry price: {entry_price}")
            return False
            
        if stop_loss_price <= 0.0:
            logger.error(f"Invalid stop loss price: {stop_loss_price}")
            return False
            
        # Check if entry and stop loss are the same
        if abs(entry_price - stop_loss_price) < 0.00001:
            logger.error(f"Entry price and stop loss are too close: {entry_price} vs {stop_loss_price}")
            return False
            
        return True
    
    def calculate_risk_amount(
        self,
        account_balance: float,
        risk_percentage: float
    ) -> float:
        """Calculate risk amount based on account balance and risk percentage."""
        try:
            risk_amount = account_balance * risk_percentage
            max_risk_amount = account_balance * self.max_risk_per_trade
            return min(risk_amount, max_risk_amount)
        except Exception as e:
            logger.error(f"Error calculating risk amount: {str(e)}")
            return 0.0
    
    def calculate_daily_risk(
        self,
        account_balance: float,
        open_trades: List[Dict[str, Any]],
        pending_trades: List[Dict[str,Any]]
    ) -> float:
        """Calculate total daily risk including open and pending trades."""
        try:
            # Calculate risk for open trades
            current_date = datetime.now(UTC).date()
            open_trades_risk = 0.0
            
            for t in open_trades:
                timestamp = t.get('timestamp')
                if isinstance(timestamp, datetime) and timestamp.date() == current_date:
                    entry_price = t.get('entry_price', 0.0)
                    stop_loss = t.get('stop_loss', 0.0)
                    position_size = t.get('position_size', 0.0)
                    open_trades_risk += abs(entry_price - stop_loss) * position_size * 100000
            
            # Calculate risk for pending trades
            pending_trades_risk = 0.0
            
            for t in pending_trades:
                timestamp = t.get('timestamp')
                if isinstance(timestamp, datetime) and timestamp.date() == current_date:
                    entry_price = t.get('entry_price', 0.0)
                    stop_loss = t.get('stop_loss', 0.0)
                    position_size = t.get('position_size', 0.0)
                    pending_trades_risk += abs(entry_price - stop_loss) * position_size * 100000
            
            return open_trades_risk + pending_trades_risk
        except Exception as e:
            logger.error(f"Error calculating daily risk: {str(e)}")
            return 0.0
    
    def can_open_new_trade(self, current_trades: List[Dict[str, Any]]) -> bool:
        """Check if a new trade can be opened based on maximum concurrent trades limit."""
        try:
            return len(current_trades) < self.max_concurrent_trades
        except Exception as e:
            logger.error(f"Error checking if can open new trade: {str(e)}")
            return False
    
    def calculate_trailing_stop(
        self,
        trade: Dict[str, Any],
        current_price: float,
        current_atr: Optional[float] = None,
        market_condition: str = 'normal'
    ) -> Tuple[bool, float]:
        """
        Calculate trailing stop level based on current price and market conditions.
        
        Args:
            trade: Trade details including entry, direction and existing stop
            current_price: Current market price
            current_atr: Current ATR value (optional)
            market_condition: Market condition ('normal', 'volatile', 'trending')
            
        Returns:
            Tuple of (should_update, new_stop_level)
        """
        # Get trade details
        entry_price = trade.get('entry_price', 0)
        current_stop = trade.get('stop_loss', 0)
        direction = trade.get('direction', '')
        
        # Calculate profit in terms of R multiple
        if direction == 'long':
            risk = entry_price - current_stop
            current_reward = current_price - entry_price
        else:  # short
            risk = current_stop - entry_price
            current_reward = entry_price - current_price
            
        if risk <= 0:
            logger.warning(f"Invalid risk value: {risk} for trade: {trade}")
            return False, current_stop
            
        r_multiple = current_reward / risk
        
        # Get trailing factor based on R multiple and market condition
        trail_factor = self.get_trail_factor(r_multiple, market_condition)
        
        # If no ATR is provided, we'll use a percentage-based trail
        if current_atr is None:
            # If trade object contains a DataFrame, calculate ATR
            if 'df' in trade and isinstance(trade['df'], pd.DataFrame):
                current_atr = self.indicators.calculate_atr(trade['df']).iloc[-1]
            else:
                # Use a percentage of price as fallback
                current_atr = current_price * 0.001  # 0.1% of price
        
        # Ensure current_atr is a valid number
        atr_value = current_atr if current_atr is not None else (current_price * 0.001)
        
        # Calculate new stop level
        if direction == 'long':
            # For long trades, move stop up
            new_stop = current_price - (atr_value * trail_factor)
            
            # Only update if the new stop is higher than the current stop
            should_update = new_stop > current_stop
        else:
            # For short trades, move stop down
            new_stop = current_price + (atr_value * trail_factor)
            
            # Only update if the new stop is lower than the current stop
            should_update = new_stop < current_stop
            
        return should_update, new_stop
    
    def calculate_stop_loss(self, entry_price: float, 
                          direction: str,
                          atr_value: Optional[float] = None,
                          pattern_type: Optional[str] = None,
                          recent_swing: Optional[Dict] = None,
                          custom_atr_multiplier: float = 1.5) -> Dict:
        """
        Calculate stop loss level based on multiple methods.
        
        Args:
            entry_price: Entry price for the trade
            direction: Trade direction ('long' or 'short')
            atr_value: Optional pre-calculated ATR value
            pattern_type: Optional pattern type for pattern-based stops
            recent_swing: Optional recent swing point for structure-based stops
            custom_atr_multiplier: Custom multiplier for ATR-based stop
            
        Returns:
            Dict with stop loss price and type
        """
        # Initialize stops list and default values
        stops = []
        atr_stop = None
        structure_stop = None
        pattern_stop = None
        
        # ATR-based stop loss (Volatility)
        if atr_value is not None:
            atr_multiplier = self._get_atr_multiplier(volatility_state='normal', market_condition='normal')
            if direction == 'long':
                atr_stop = entry_price - (atr_value * atr_multiplier * custom_atr_multiplier)
            else:
                atr_stop = entry_price + (atr_value * atr_multiplier * custom_atr_multiplier)
            stops.append(atr_stop)
        
        # Structure-based stop loss
        if recent_swing is not None:
            if direction == 'long' and 'low' in recent_swing:
                # For long positions, use recent swing low
                structure_stop = recent_swing['low'] - (atr_value * 0.5 if atr_value else 0.0001)
            elif direction == 'short' and 'high' in recent_swing:
                # For short positions, use recent swing high
                structure_stop = recent_swing['high'] + (atr_value * 0.5 if atr_value else 0.0001)
            
            if structure_stop:
                stops.append(structure_stop)
        
        # Pattern-based stop loss
        if pattern_type:
            # Different patterns have different optimal stop placements
            if pattern_type == 'double_bottom' and direction == 'long':
                pattern_stop = entry_price - (entry_price * 0.01)  # 1% below entry
            elif pattern_type == 'double_top' and direction == 'short':
                pattern_stop = entry_price + (entry_price * 0.01)  # 1% above entry
            elif pattern_type == 'triangle' or pattern_type == 'wedge':
                pattern_stop = entry_price - (atr_value * 2 if atr_value else entry_price * 0.02) if direction == 'long' else entry_price + (atr_value * 2 if atr_value else entry_price * 0.02)
            else:
                # Default pattern stop
                pattern_stop = entry_price - (entry_price * 0.02) if direction == 'long' else entry_price + (entry_price * 0.02)
            
            if pattern_stop:
                stops.append(pattern_stop)
        
        # Ensure we have at least one stop level
        if not stops:
            # Default stop loss (fixed percentage)
            default_stop = entry_price - (entry_price * 0.02) if direction == 'long' else entry_price + (entry_price * 0.02)
            stops.append(default_stop)
        
        # Select the most appropriate stop loss
        # For long trades: choose the highest (closest to entry) WHILE STILL BELOW ENTRY
        # For short trades: choose the lowest (closest to entry) WHILE STILL ABOVE ENTRY
        if direction == 'long' or direction == 'BUY':
            # Filter out stops that are at or above entry price
            valid_stops = [stop for stop in stops if stop < entry_price]
            if valid_stops:
                selected_stop = max(valid_stops)  # Highest valid stop (closest to entry) for long
            else:
                # Fallback if no valid stops: 2% below entry
                selected_stop = entry_price * 0.98
        else:  # direction == 'short' or direction == 'SELL'
            # Filter out stops that are at or below entry price
            valid_stops = [stop for stop in stops if stop > entry_price]
            if valid_stops:
                selected_stop = min(valid_stops)  # Lowest valid stop (closest to entry) for short
            else:
                # Fallback if no valid stops: 2% above entry
                selected_stop = entry_price * 1.02
        
        return {
            'price': selected_stop,
            'type': self._determine_stop_type(selected_stop, atr_stop, structure_stop, pattern_stop),
            'atr_stop': atr_stop,
            'structure_stop': structure_stop,
            'pattern_stop': pattern_stop
        }
        
    def _determine_stop_type(self, selected_stop, atr_stop, structure_stop, pattern_stop):
        """
        Determine the type of stop loss being used.
        
        Args:
            selected_stop: The selected stop loss price
            atr_stop: ATR-based stop price
            structure_stop: Structure-based stop price
            pattern_stop: Pattern-based stop price
            
        Returns:
            String indicating the type of stop loss
        """
        if selected_stop == atr_stop:
            return "volatility"
        elif selected_stop == structure_stop:
            return "structure"
        elif selected_stop == pattern_stop:
            return "pattern"
        else:
            return "default"

    def validate_trade(self, trade: Dict, account_balance: float, 
                     open_trades: List[Dict], 
                     correlation_matrix: Optional[Dict] = None) -> Dict:
        """
        Validate a trade against risk management rules.
        
        Args:
            trade: Trade details including entry, stop, and direction
            account_balance: Current account balance
            open_trades: List of currently open trades
            correlation_matrix: Optional correlation matrix for instruments
            
        Returns:
            Dict with validation result and reason
        """
        # Check max number of open trades
        if len(open_trades) >= self.max_concurrent_trades:
            return {
                'valid': False,
                'reason': f"Max number of open trades ({self.max_concurrent_trades}) reached"
            }
        
        # Check daily risk
        if self.daily_stats['total_risk'] >= self.max_daily_risk * account_balance:
            return {
                'valid': False,
                'reason': f"Max daily risk reached ({self.max_daily_risk*100}% of account)"
            }
        
        # Check correlation risk
        if correlation_matrix is not None and trade.get('symbol') in correlation_matrix:
            correlated_count = 0
            for open_trade in open_trades:
                if open_trade.get('symbol') in correlation_matrix:
                    correlation = correlation_matrix.get(
                        trade.get('symbol'), {}).get(open_trade.get('symbol'), 0)
                    if correlation >= self.correlation_limits['max_correlation']:
                        correlated_count += 1
                    
            if correlated_count >= self.max_concurrent_trades:
                return {
                    'valid': False,
                    'reason': f"Max correlated trades ({self.max_concurrent_trades}) reached"
                }
            
        # Get trade parameters
        symbol = trade.get('symbol', '')
        entry = trade.get('entry_price', 0) or trade.get('entry', 0)
        stop = trade.get('stop_loss', 0)
        requested_size = trade.get('position_size', 0)
        
        # Determine position size based on config
        if self.use_fixed_lot_size:
            # Use fixed lot size from config
            position_size = min(self.fixed_lot_size, self.max_lot_size)
            logger.info(f"Using fixed lot size from config: {position_size}")
        elif requested_size > 0:
            # Use requested size but cap at max lot size
            position_size = min(requested_size, self.max_lot_size)
            logger.info(f"Using requested position size (capped): {position_size}")
        else:
            # Calculate position size based on risk
            try:
                position_size = self.calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=self.max_risk_per_trade * 100,  # Convert to percentage
                    entry_price=entry,
                    stop_loss_price=stop,
                    symbol=symbol
                )
                logger.info(f"Calculated position size based on risk: {position_size}")
            except Exception as e:
                logger.error(f"Error calculating position size: {str(e)}")
                position_size = min(0.01, self.max_lot_size)  # Fallback to minimum
        
        # Validate risk per trade if not using fixed lot size
        if not self.use_fixed_lot_size and stop != 0 and entry != 0:
            risk_amount = abs(entry - stop) * position_size
            risk_percentage = risk_amount / account_balance
            
            if risk_percentage > self.max_risk_per_trade:
                # Try to adjust position size to meet risk requirement
                adjusted_position = self.max_risk_per_trade * account_balance / abs(entry - stop)
                adjusted_position = round(adjusted_position, 2)  # Round to standard lot precision
                
                # Only adjust if it's a meaningful adjustment
                if adjusted_position >= 0.01 and adjusted_position < position_size:
                    logger.warning(f"Position size reduced from {position_size} to {adjusted_position} due to risk limits")
                    position_size = adjusted_position
                else:
                    return {
                        'valid': False,
                        'reason': f"Trade risk ({risk_percentage*100:.2f}%) exceeds max risk per trade ({self.max_risk_per_trade*100}%) and cannot be adjusted"
                    }
        
        # Success result with adjusted position size
        return {
            'valid': True,
            'reason': "Trade meets all risk management criteria",
            'adjusted_position_size': position_size
        }
    
    def calculate_drawdown(self) -> float:
        """
        Calculate current drawdown based on peak balance versus current balance.
        
        Returns:
            float: Current drawdown as a percentage (0.0 to 1.0)
        """
        try:
            account_info = self._get_account_info()
            if not account_info:
                logger.error("Failed to get account info for drawdown calculation")
                return 0.0
            
            current_balance = account_info.get('balance', 0.0)
            equity = account_info.get('equity', current_balance)
            
            # Calculate drawdown from peak balance
            peak_balance = max(current_balance, self.daily_stats.get('starting_balance', current_balance))
            absolute_drawdown = peak_balance - equity
            drawdown_percentage = (absolute_drawdown / peak_balance) if peak_balance > 0 else 0.0
            
            logger.debug(f"Drawdown calculation: Peak={peak_balance}, Current={equity}, DD%={drawdown_percentage*100:.2f}%")
            return drawdown_percentage
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        signal_type: str,
        market_condition: str = 'normal',
        risk_reward_ratio: Optional[float] = None
    ) -> Tuple[List[Dict[str, float]], float]:
        """
        Calculate take profit levels based on market conditions and risk-reward ratio.
        """
        try:
            # Calculate base risk
            risk = abs(entry_price - stop_loss)
            
            # Get appropriate TP levels based on market condition
            if market_condition in self.dynamic_tp_levels:
                tp_levels = self.dynamic_tp_levels[market_condition]
            else:
                # Default to ranging market TP levels
                tp_levels = self.dynamic_tp_levels['ranging']
            
            # Override with single R:R ratio if provided
            if risk_reward_ratio is not None:
                tp_levels = [{
                    'size': 1.0,
                    'ratio': risk_reward_ratio
                }]
            
            # Calculate take profit levels
            take_profits = []
            weighted_tp = 0.0
            total_size = 0.0
            
            for level in tp_levels:
                tp_distance = risk * level['ratio']
                tp_price = (
                    entry_price + tp_distance if signal_type == "BUY"
                    else entry_price - tp_distance
                )
                
                take_profits.append({
                    'size': level['size'],
                    'price': round(tp_price, 5),
                    'r_multiple': level['ratio']
                })
                
                weighted_tp += tp_price * level['size']
                total_size += level['size']
            
            # Calculate final take profit as weighted average
            final_tp = weighted_tp / total_size if total_size > 0 else (
                entry_price + risk if signal_type == "BUY"
                else entry_price - risk
            )
            
            logger.info(f"Calculated {len(take_profits)} take profit levels for {signal_type}")
            for tp in take_profits:
                logger.debug(f"TP Level: {tp['size']*100}% at {tp['price']:.5f} ({tp['r_multiple']}R)")
            
            return take_profits, round(final_tp, 5)
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            # Return default 1.5:1 R:R ratio
            calculated_risk = abs(entry_price - stop_loss)
            default_tp = (
                entry_price + (calculated_risk * 1.5) if signal_type == "BUY"
                else entry_price - (calculated_risk * 1.5)
            )
            return ([{'size': 1.0, 'price': round(default_tp, 5), 'r_multiple': 1.5}],
                    round(default_tp, 5))
    
    def update_open_trades(self, trades: List[Dict]):
        """Update the list of open trades."""
        self.open_trades = trades
    

    def validate_trade_risk(
        self,
        account_balance: float,
        risk_amount: float,
        current_daily_risk: float,
        current_weekly_risk: float,
        daily_trades: int,
        weekly_trades: int,
        current_drawdown: float,
        consecutive_losses: int,
        last_trade_time: Optional[datetime] = None,
        correlations: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Validate if a trade can be taken based on all risk parameters.
        """
        try:
            # 1. Check risk limits
            if risk_amount > account_balance * self.max_risk_per_trade:
                return False, f"Risk amount {risk_amount:.2f} exceeds max risk per trade"
            
            total_daily_risk = current_daily_risk + risk_amount
            if total_daily_risk > account_balance * self.max_daily_risk:
                return False, f"Total daily risk {total_daily_risk:.2f} would exceed limit"
            
            total_weekly_risk = current_weekly_risk + risk_amount
            if total_weekly_risk > account_balance * self.max_weekly_loss:
                return False, f"Total weekly risk {total_weekly_risk:.2f} would exceed limit"
    
            # 2. Check trade frequency limits
            if daily_trades >= self.max_daily_trades:
                    return False, f"Daily trade limit {self.max_daily_trades} reached"
                
            if weekly_trades >= self.max_weekly_trades:
                    return False, f"Weekly trade limit {self.max_weekly_trades} reached"
                
                # 3. Check drawdown limits
            if current_drawdown >= self.max_drawdown_pause:
                return False, f"Max drawdown {self.max_drawdown_pause*100}% reached"
                
                # 4. Check consecutive losses
            if consecutive_losses >= self.consecutive_loss_limit:
                    return False, f"Consecutive loss limit {self.consecutive_loss_limit} reached"

            # 5. Check trade spacing
            if last_trade_time:
                time_since_last = datetime.now(UTC) - last_trade_time
                min_spacing = timedelta(hours=self.min_trades_spacing)
                if time_since_last < min_spacing:
                    return False, f"Minimum trade spacing {self.min_trades_spacing}h not met"

            # 6. Check correlations
            if correlations:
                for symbol, corr in correlations.items():
                    if abs(corr) > self.correlation_limits['max_correlation']:
                        return False, f"Correlation with {symbol} ({corr:.2f}) exceeds limit"

            # All checks passed
            return True, "Trade validated"

        except Exception as e:
            logger.error(f"Error validating trade risk: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_stats = {
            'total_risk': 0.0,
            'realized_pnl': 0.0,
            'trade_count': 0,
            'starting_balance': 0.0,
            'last_reset': datetime.now(UTC).date()
        }

    def update_daily_performance(self, trade_result: Dict) -> Dict:
        """
        Update daily performance tracking with trade result.
        
        Args:
            trade_result: Dictionary with trade result details
            
        Returns:
            Dictionary with updated performance metrics
        """
        profit_loss = trade_result.get('profit_loss', 0)
        
        if profit_loss < 0:
            self.daily_stats['realized_pnl'] -= abs(profit_loss)
        else:
            self.daily_stats['realized_pnl'] += profit_loss
            
        # Increment trade count
        self.daily_stats['trade_count'] += 1
        
        # Ensure daily stats are up-to-date
        current_date = datetime.now(UTC).date()
        if self.daily_stats['last_reset'] < current_date:
            self.reset_daily_stats()
            
        return {
            'daily_losses': abs(min(0, self.daily_stats['realized_pnl'])),
            'daily_wins': max(0, self.daily_stats['realized_pnl']),
            'net_daily_pnl': self.daily_stats['realized_pnl'],
            'trade_count': self.daily_stats['trade_count']
        }

    def update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Update correlation matrix between trading instruments.
        
        Args:
            price_data: Dictionary of price DataFrames by symbol
            
        Returns:
            Updated correlation matrix
        """
        symbols = list(price_data.keys())
        matrix = {}
        
        for symbol1 in symbols:
            if symbol1 not in matrix:
                matrix[symbol1] = {}
                
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = 1.0
                    continue
                    
                # Calculate correlation between closing prices with error handling
                try:
                    # Get DataFrames
                    df1 = price_data[symbol1].copy()
                    df2 = price_data[symbol2].copy()
                    
                    # Check if close columns exist
                    if 'close' not in df1.columns or 'close' not in df2.columns:
                        logger.warning(f"Missing 'close' column for {symbol1} or {symbol2}")
                        matrix[symbol1][symbol2] = 0.0
                        continue
                    
                    # Convert to float to ensure numeric computation
                    s1 = pd.Series(df1['close'].values).astype(float)
                    s2 = pd.Series(df2['close'].values).astype(float)
                    
                    # Calculate correlation safely
                    if len(s1) > 1 and len(s2) > 1:
                        correlation = s1.corr(s2)
                        # Handle NaN result
                        if pd.isna(correlation):
                            matrix[symbol1][symbol2] = 0.0
                        else:
                            matrix[symbol1][symbol2] = float(correlation)
                    else:
                        logger.warning(f"Not enough data for correlation between {symbol1} and {symbol2}")
                        matrix[symbol1][symbol2] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {str(e)}")
                    matrix[symbol1][symbol2] = 0.0
                
        # Store the correlation matrix in the instance
        self.correlation_matrix = matrix
        return matrix

    def get_trail_factor(self, r_multiple: float, market_condition: str = 'normal') -> float:
        """
        Calculate trailing stop factor based on R-multiple and market condition.
        
        Args:
            r_multiple: Current R-multiple (profit / initial risk)
            market_condition: Current market condition
            
        Returns:
            float: Trailing stop factor
        """
        try:
            # Adjust trail factor based on R-multiple achieved
            if r_multiple >= 2.0:
                base_factor = 1.0  # Tight trail at 2R+
            elif r_multiple >= 1.5:
                base_factor = 1.5  # Medium trail at 1.5R+
            elif r_multiple >= 1.0:
                base_factor = 2.0  # Wider trail at 1R+
            else:
                base_factor = 2.5  # Very wide trail below 1R
            
            # Adjust for market condition
            if market_condition == 'trending':
                base_factor *= 1.5  # Wider trails in trending market
            elif market_condition == 'ranging':
                base_factor *= 0.75  # Tighter trails in ranging market
            elif market_condition == 'choppy':
                base_factor *= 0.5  # Even tighter trails in choppy market
            
            return base_factor
            
        except Exception as e:
            logger.error(f"Error calculating trail factor: {str(e)}")
            return 2.0  # Default to conservative trail factor
    
    def calculate_pip_value(self, symbol: str, price: float) -> float:
        """
        Calculate the value of one pip for the given symbol at the current price.
        
        Args:
            symbol: The trading symbol
            price: Current price of the symbol
            
        Returns:
            float: The value of one pip in account currency
        """
        try:
            # Get symbol information from MT5
            symbol_info = self.mt5_handler.get_symbol_info(symbol)
            
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                # Fallback calculation
                if symbol.endswith('JPY') or 'JPY' in symbol:
                    pip_size = 0.01
                elif symbol.startswith('XAU'):
                    pip_size = 0.1
                elif symbol.startswith('XAG'):
                    pip_size = 0.01
                # Special handling for cryptocurrency pairs
                elif symbol.endswith('USDm') or symbol.endswith('USDT') or symbol.endswith('USD') and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                    pip_size = 1.0
                else:
                    pip_size = 0.0001  # Default for most forex pairs
                return 0.1  # Return a default conversion factor
            
            # Determine pip size based on digits
            digits = symbol_info.digits
            
            # Special handling for cryptocurrency pairs regardless of digits
            if symbol.endswith('USDm') or symbol.endswith('USDT') or symbol.endswith('USD') and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                pip_size = 1.0  # For cryptocurrencies, define 1 unit as 1 pip
            elif digits == 3 or digits == 5:
                pip_size = 0.0001  # 4-digit pricing (standard forex)
            elif digits == 2:
                pip_size = 0.01    # 2-digit pricing (JPY pairs)
            elif digits == 1:
                pip_size = 0.1     # 1-digit pricing
            elif digits == 0:
                pip_size = 1.0     # 0-digit pricing
            else:
                pip_size = 0.0001  # Default to standard forex pip size
            
            # Get contract size (standard lot is typically 100,000 units)
            contract_size = symbol_info.trade_contract_size
            
            # The pip value calculation depends on the account currency
            # For simplicity, we're assuming account currency is USD
            # For USD-based account:
            
            # Case 1: USD is the quote currency (e.g., EUR/USD)
            if symbol.endswith('USD'):
                pip_value = pip_size * contract_size
            
            # Case 2: USD is the base currency (e.g., USD/JPY)
            elif symbol.startswith('USD'):
                pip_value = (pip_size * contract_size) / price
            
            # Case 3: Neither currency is USD (e.g., EUR/GBP)
            # For this, we would need to know the USD/quote_currency rate
            # For simplicity, we'll use a default calculation
            else:
                pip_value = pip_size * 10  # Simplified assumption
            
            logger.debug(f"Calculated pip value for {symbol}: {pip_value}")
            return pip_value
            
        except Exception as e:
            logger.error(f"Error calculating pip value: {str(e)}")
            # Fallback to a reasonable default if error occurs
            return 0.1  # Default conversion factor

    def get_account_history(self, days: int = 1) -> Optional[List[Dict]]:
        """
        Get account trading history for the specified number of days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Optional[List[Dict]]: List of trade history records or None if error
        """
        try:
            if not self.mt5_handler:
                logger.error("MT5Handler not initialized")
                return None
                
            # Calculate time range
            now = datetime.now()
            from_date = now - timedelta(days=days)
            
            # Get history from MT5
            history = None
            if hasattr(self.mt5_handler, 'get_history_deals'):
                # Type hint for get_history_deals
                history = self.mt5_handler.get_history_deals(from_date, now)  # type: ignore
                
            if not history:
                logger.warning(f"No history deals found for the last {days} days")
                return []
                
            # Process history records
            processed_history = []
            for deal in history:
                processed_history.append({
                    "ticket": deal.ticket,
                    "time": datetime.fromtimestamp(deal.time) if hasattr(deal, 'time') else now,
                    "symbol": deal.symbol,
                    "type": deal.type,
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit_loss": deal.profit,
                    "commission": deal.commission,
                    "swap": deal.swap
                })
                
            logger.info(f"Retrieved {len(processed_history)} history records for the last {days} days")
            return processed_history
            
        except Exception as e:
            logger.error(f"Error getting account history: {str(e)}")
            return None

    def check_risk_limit(self, max_daily_loss_pct: float = 2.0) -> bool:
        """
        Check if daily loss limit is exceeded.
        
        Args:
            max_daily_loss_pct: Maximum allowed daily loss as percentage of balance
            
        Returns:
            bool: True if within limits, False if exceeded
        """
        try:
            # Get account information
            account_info = self._get_account_info()
            if not account_info or 'balance' not in account_info:
                logger.error("Failed to get account info")
                return False
                
            balance = account_info['balance']
            
            # Get today's trading history
            history = self.get_account_history(days=1)
            if history is None:  # Error occurred
                logger.error("Failed to get account history")
                return False
                
            # If no trades today, we're within limits
            if not history:
                return True
                
            # Calculate total P&L for today
            daily_pl = sum(deal['profit_loss'] for deal in history)
            
            # Check if we're in a loss
            if daily_pl < 0:
                loss_pct = (abs(daily_pl) / balance) * 100
                
                if loss_pct > max_daily_loss_pct:
                    logger.error(f"Daily loss limit exceeded: {loss_pct:.2f}% > {max_daily_loss_pct}%")
                    return False
                    
                logger.info(f"Current daily loss: {loss_pct:.2f}% (limit: {max_daily_loss_pct}%)")
            else:
                logger.info(f"Current daily P&L: +{daily_pl:.2f}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limit: {str(e)}")
            return False

    def _get_atr_multiplier(self, volatility_state: str, market_condition: str) -> float:
        """
        Calculate ATR multiplier based on volatility state and market condition.
        
        Args:
            volatility_state: Current volatility state ('low', 'normal', 'high')
            market_condition: Current market condition ('trending', 'ranging', etc.)
            
        Returns:
            float: Calculated ATR multiplier
        """
        try:
            # Adjust ATR multiplier based on volatility state
            if volatility_state == 'high':
                base_multiplier = 1.3
            elif volatility_state == 'low':
                base_multiplier = 0.75
            else:
                base_multiplier = 1.0
            
            # Adjust for market condition
            if market_condition == 'trending':
                base_multiplier *= 1.2
            elif market_condition == 'choppy':
                base_multiplier *= 0.8
            
            return base_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating ATR multiplier: {str(e)}")
            return 1.0  # Default to conservative multiplier

    def initialize(self, config=None, timeframe=None):
        """
        Initialize or reinitialize the RiskManager with the given config.
        
        Args:
            config: Configuration object that contains risk parameters
            timeframe: Optional timeframe to use for loading timeframe-specific parameters
        """
        if timeframe:
            # Update the current timeframe if provided
            self.timeframe = timeframe
            logger.info(f"Updating RiskManager timeframe to {timeframe}")
            
            # Load timeframe-specific parameters
            try:
                from config.config import get_risk_config
                timeframe_config = get_risk_config(timeframe)
                
                # Update core risk parameters with timeframe-specific values
                self.max_risk_per_trade = timeframe_config.get('max_risk_per_trade', self.max_risk_per_trade)
                self.max_daily_loss = timeframe_config.get('max_daily_loss', self.max_daily_loss)
                
                # Update position management with timeframe-specific values
                self.max_concurrent_trades = timeframe_config.get('max_concurrent_trades', self.max_concurrent_trades)
                self.max_daily_trades = timeframe_config.get('max_daily_trades', self.max_daily_trades)
                self.min_trades_spacing = timeframe_config.get('min_trades_spacing', self.min_trades_spacing)
                
                # Update correlation threshold
                if 'correlation_threshold' in timeframe_config:
                    self.correlation_limits['max_correlation'] = timeframe_config['correlation_threshold']
        
                
                logger.info(f"RiskManager updated with {timeframe} timeframe parameters")
            except Exception as e:
                logger.error(f"Error loading timeframe-specific parameters: {e}")
        
        if not config:
            logger.info("No custom config provided for RiskManager, using current settings")
            return
            
        # Update the configuration if provided
        try:
            # Get risk config from the main config
            risk_config = config.get('RISK_CONFIG', {})
            if not risk_config:
                logger.info("No risk configuration found in provided config, using current settings")
                return
                
            # Update internal config
            for key, value in risk_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"Updated RiskManager config: {key} = {value}")
                    
            logger.info("RiskManager initialized with custom configuration")
            
        except Exception as e:
            logger.error(f"Error initializing RiskManager: {str(e)}")
            logger.info("Using current RiskManager configuration")

    def check_max_daily_loss(self, current_daily_loss: float) -> bool:
        """
        Check if the current daily loss exceeds the maximum allowed.
        
        Args:
            current_daily_loss: Current daily loss amount
            
        Returns:
            bool: True if trading should continue, False if max loss exceeded
        """
        logger.debug(f"Checking max daily loss for timeframe {self.timeframe}: Current: {current_daily_loss}, Max: {self.max_daily_loss}")
        if current_daily_loss >= self.max_daily_loss:
            logger.warning(f"Maximum daily loss threshold reached: {current_daily_loss} >= {self.max_daily_loss}")
            return False
        return True

    def check_max_drawdown(self, current_drawdown: float) -> bool:
        """
        Check if the current drawdown exceeds the maximum allowed.
        
        Args:
            current_drawdown: Current drawdown percentage (0.0 to 1.0)
            
        Returns:
            bool: True if trading should continue, False if max drawdown exceeded
        """
        logger.debug(f"Checking max drawdown for timeframe {self.timeframe}: Current: {current_drawdown}, Max: {self.max_drawdown}")
        if current_drawdown >= self.max_drawdown:
            logger.warning(f"Maximum drawdown threshold reached: {current_drawdown} >= {self.max_drawdown}")
            return False
        return True

    def check_max_concurrent_trades(self, current_open_trades: int) -> bool:
        """
        Check if the current number of open trades exceeds the maximum allowed.
        
        Args:
            current_open_trades: Current number of open trades
            
        Returns:
            bool: True if more trades allowed, False if max reached
        """
        logger.debug(f"Checking max concurrent trades for timeframe {self.timeframe}: Current: {current_open_trades}, Max: {self.max_concurrent_trades}")
        if current_open_trades >= self.max_concurrent_trades:
            logger.warning(f"Maximum concurrent trades reached: {current_open_trades} >= {self.max_concurrent_trades}")
            return False
        return True
        
    def check_max_daily_trades(self, current_daily_trades: int) -> bool:
        """
        Check if the current number of daily trades exceeds the maximum allowed.
        
        Args:
            current_daily_trades: Current number of trades today
            
        Returns:
            bool: True if more trades allowed, False if max reached
        """
        logger.debug(f"Checking max daily trades for timeframe {self.timeframe}: Current: {current_daily_trades}, Max: {self.max_daily_trades}")
        if current_daily_trades >= self.max_daily_trades:
            logger.warning(f"Maximum daily trades reached: {current_daily_trades} >= {self.max_daily_trades}")
            return False
        return True
        
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float,
        symbol: str,
        market_condition: str = 'normal'
    ) -> float:
        """
        Calculate position size based on risk parameters and configuration.
        
        Args:
            account_balance: Account balance
            risk_per_trade: Risk percentage per trade (0-100)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            market_condition: Market condition
            
        Returns:
            float: Calculated position size in lots
        """
        try:
            # First check if we're using fixed lot size from config
            if self.use_fixed_lot_size:
                # Use fixed lot size from config
                logger.info(f"Using fixed lot size of {self.fixed_lot_size} from config")
                return min(self.fixed_lot_size, self.max_lot_size)
                
            # If not using fixed lot size, calculate based on risk
            # Validate inputs first
            if not self._validate_position_inputs(account_balance, risk_per_trade/100, entry_price, stop_loss_price):
                logger.warning("Invalid position sizing inputs, using default size")
                return min(0.01, self.max_lot_size)  # Use minimum default
                
            # Adjust risk percentage based on market condition
            adjusted_risk = risk_per_trade
            if market_condition in self.market_condition_adjustments:
                adjusted_risk *= self.market_condition_adjustments[market_condition]
                logger.debug(f"Adjusted risk by factor {self.market_condition_adjustments[market_condition]} for {market_condition} market")
                
            # Calculate risk amount
            risk_amount = account_balance * (adjusted_risk / 100)
            
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss_price)
            if stop_distance <= 0:
                logger.error(f"Invalid stop distance: {stop_distance}")
                return min(0.01, self.max_lot_size)  # Use minimum default
                
            # Calculate risk per pip
            risk_per_pip = risk_amount / stop_distance
            
            # Calculate position size based on risk
            position_size = risk_per_pip * 0.0001  # Convert to lots
            
            # If we have MT5Handler, use it for more accurate calculation
            if self.mt5_handler and hasattr(self.mt5_handler, 'calculate_position_size'):
                # Let MT5Handler do the calculation with risk parameters
                try:
                    return min(self.mt5_handler.calculate_position_size(
                        symbol=symbol,
                        price=entry_price,
                        risk_percent=adjusted_risk,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss_price
                    ), self.max_lot_size)
                except Exception as e:
                    logger.error(f"Error in MT5Handler position sizing: {str(e)}")
                    # Fall through to our backup calculation
            
            # Ensure position size doesn't exceed max
            position_size = min(position_size, self.max_lot_size)
            
            # Ensure minimum position size
            position_size = max(position_size, 0.01)
            
            # Round to 2 decimal places (standard lot precision)
            position_size = round(position_size, 2)
            
            logger.info(f"Calculated position size: {position_size} lots based on risk")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            # Return minimum position size on error
            return min(0.01, self.max_lot_size)
        
    def update_timeframe(self, timeframe: str) -> None:
        """
        Update the risk manager parameters for a new timeframe.
        
        Args:
            timeframe: New timeframe to use (M1, M5, M15, H1, H4, D1)
        """
        if timeframe != self.timeframe:
            logger.info(f"Updating risk parameters for timeframe: {timeframe}")
            
            # Store new timeframe
            self.timeframe = timeframe
            
            # Load timeframe-specific parameters
            try:
                from config.config import get_risk_config
                timeframe_config = get_risk_config(timeframe)
                
                # Update core risk parameters with timeframe-specific values
                self.max_risk_per_trade = timeframe_config.get('max_risk_per_trade', self.max_risk_per_trade)
                self.max_daily_loss = timeframe_config.get('max_daily_loss', self.max_daily_loss)
                
                # Update position management
                self.max_concurrent_trades = timeframe_config.get('max_concurrent_trades', self.max_concurrent_trades)
                self.max_daily_trades = timeframe_config.get('max_daily_trades', self.max_daily_trades)
                self.max_weekly_trades = timeframe_config.get('max_weekly_trades', self.max_weekly_trades)
                self.min_trades_spacing = timeframe_config.get('min_trades_spacing', self.min_trades_spacing)
                
                logger.info(f"Updated risk parameters: max_risk_per_trade={self.max_risk_per_trade}, max_daily_trades={self.max_daily_trades}")
            except Exception as e:
                logger.error(f"Error updating risk parameters for timeframe {timeframe}: {str(e)}")
        else:
            logger.debug(f"Timeframe unchanged: {timeframe}, no risk parameter updates needed")
        
    # Add market data integration
    def update_market_data(self, symbol, timeframe, data):
        """
        Update internal market data for risk calculations.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: Market data DataFrame
        """
        if not hasattr(self, 'market_data'):
            self.market_data = {}
            
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
            
        self.market_data[symbol][timeframe] = data
        
        # Recalculate volatility metrics
        self._calculate_volatility_metrics(symbol, timeframe)
        
    def _calculate_volatility_metrics(self, symbol, timeframe):
        """
        Calculate volatility metrics like ATR.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        if not hasattr(self, 'market_data') or symbol not in self.market_data or timeframe not in self.market_data[symbol]:
            return
            
        data = self.market_data[symbol][timeframe]
        
        try:
            # Calculate simple ATR
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate true range
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate 14-period ATR
            atr = np.mean(true_range[-14:]) if len(true_range) >= 14 else None
            
            # Store for later use
            if not hasattr(self, 'volatility_metrics'):
                self.volatility_metrics = {}
                
            if symbol not in self.volatility_metrics:
                self.volatility_metrics[symbol] = {}
                
            self.volatility_metrics[symbol][timeframe] = {
                'atr': atr,
                'last_updated': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol} {timeframe}: {str(e)}")
            
    def get_volatility(self, symbol, timeframe=None):
        """
        Get current volatility metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Optional specific timeframe
            
        Returns:
            ATR value or None if not available
        """
        if not hasattr(self, 'volatility_metrics') or symbol not in self.volatility_metrics:
            return None
            
        if timeframe is not None:
            if timeframe in self.volatility_metrics[symbol]:
                return self.volatility_metrics[symbol][timeframe].get('atr')
            return None
            
        # Return the first available timeframe's ATR if no specific timeframe requested
        for tf in self.volatility_metrics[symbol]:
            return self.volatility_metrics[symbol][tf].get('atr')
            
        return None
        