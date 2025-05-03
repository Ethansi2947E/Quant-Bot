import time
from datetime import datetime, timedelta, UTC
import math
from typing import Dict, List, Optional, Union, Tuple, Any, TYPE_CHECKING
from config.config import TRADING_CONFIG, TELEGRAM_CONFIG, RISK_MANAGER_CONFIG
from loguru import logger
import pandas as pd
import MetaTrader5 as mt5
import numpy as np
import traceback
import asyncio

# Use TYPE_CHECKING for import that's only used for type hints
if TYPE_CHECKING:
    from src.mt5_handler import MT5Handler
from src.utils.indicators import calculate_atr
from src.utils.market_utils import calculate_pip_value, convert_pips_to_price

# Singleton instance for global reference
_risk_manager_instance = None

class RiskManager:
    """Risk manager handles position sizing, risk control, and trade management."""

    def __init__(self, mt5_handler = None):
        """
        Initialize the risk manager with a MT5 handler and configuration.
        
        Args:
            mt5_handler: MetaTrader5 interface instance (optional)
            timeframe: The timeframe to use for risk calculations, defaults to "M15"
        """
        # Singleton pattern
        global _risk_manager_instance
        
        # If an instance already exists, use it
        if _risk_manager_instance is not None:
            logger.info("Using existing RiskManager instance")
            self.__dict__ = _risk_manager_instance.__dict__
            return
            
        _risk_manager_instance = self
        
        # Initialize MT5 handler
        self.mt5_handler = mt5_handler
        if self.mt5_handler is None:
            # Defer the import to avoid circular imports
            from src.mt5_handler import MT5Handler
            self.mt5_handler = MT5Handler()
            
        self.mt5 = self.mt5_handler  # Alias for compatibility
        
        # Store timeframe
        
        # Get configuration from Risk Manager Config
        from config.config import get_risk_config
        self.config = get_risk_config()
        
        # Set risk parameters from config
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01)
        self.max_daily_risk = self.config.get('max_daily_risk', 0.03)
        self.max_drawdown_pause = self.config.get('max_drawdown_pause', 0.05)
        self.max_daily_trades = self.config.get('max_daily_trades', 8)
        self.min_trades_spacing = self.config.get('min_trades_spacing', 1)
        self.max_concurrent_trades = self.config.get('max_concurrent_trades', 2)
        # Add minimum risk:reward ratio (configurable)
        self.min_risk_reward = self.config.get('min_risk_reward', 1.0)
        
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
            
    def set_mt5_handler(self, mt5_handler) -> None:
        """
        Set the MT5Handler instance for this RiskManager.
        
        Args:
            mt5_handler: The MT5Handler instance to use
        """
        logger.info("Setting MT5Handler in RiskManager")
        self.mt5_handler = mt5_handler

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
        tp = trade.get('take_profit', 0)
        requested_size = trade.get('position_size', 0)
        
        # --- Enforce minimum risk:reward ratio ---
        risk = abs(entry - stop)
        reward = abs(tp - entry)
        if reward <= 0 or risk <= 0 or (reward / risk) < self.min_risk_reward:
            return {
                'valid': False,
                'reason': f"Risk:Reward ratio too low ({reward/risk if risk > 0 else 0:.2f}), must be at least {self.min_risk_reward:.2f}"
            }
        
        # Determine position size based on config
        if self.use_fixed_lot_size:
            # Use fixed lot size from config, but always align with symbol's constraints
            if self.mt5_handler:
                min_lot_size = self.mt5_handler.get_symbol_min_lot_size(symbol)
                normalized_lot = self.mt5_handler.normalize_volume(symbol, self.fixed_lot_size)
                position_size = max(min_lot_size, normalized_lot)
                position_size = min(position_size, self.max_lot_size)
                logger.info(f"Using fixed lot size from config (normalized): {position_size}")
            else:
                position_size = min(self.fixed_lot_size, self.max_lot_size)
                logger.info(f"Using fixed lot size from config (no mt5_handler): {position_size}")
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
                timeframe_config = get_risk_config()
                
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
        Calculate position size based on account balance, risk percentage, and stop loss distance.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage (0-100)
            entry_price: Entry price of the trade
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            market_condition: Current market condition (normal, volatile, etc.)
            
        Returns:
            float: Position size in lots
        """
        try:
            # First check if we're using fixed lot size from config
            if self.use_fixed_lot_size:
                # Use fixed lot size from config
                position_size = min(self.fixed_lot_size, self.max_lot_size)
                
                # If we have MT5Handler, ensure position size respects symbol's constraints
                if self.mt5_handler:
                    # Get minimum lot size for this symbol
                    min_lot_size = self.mt5_handler.get_symbol_min_lot_size(symbol)
                    
                    # Make sure fixed lot size is not less than symbol's minimum
                    if position_size < min_lot_size:
                        position_size = min_lot_size
                        logger.info(f"Adjusted fixed lot size to symbol's minimum: {position_size}")
                    
                    # Normalize volume according to symbol's volume_step
                    position_size = self.mt5_handler.normalize_volume(symbol, position_size)
                
                logger.info(f"Using fixed lot size of {position_size} from config")
                return position_size
                
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
            
            # Ensure position size doesn't exceed max
            position_size = min(position_size, self.max_lot_size)
            
            # If we have MT5Handler, ensure position size respects symbol's constraints
            if self.mt5_handler:
                # Get minimum lot size for this symbol
                min_lot_size = self.mt5_handler.get_symbol_min_lot_size(symbol)
                
                # Make sure position size is not less than symbol's minimum
                if position_size < min_lot_size:
                    position_size = min_lot_size
                    logger.info(f"Adjusted position size to symbol's minimum: {position_size}")
                
                # Normalize volume according to symbol's volume_step
                position_size = self.mt5_handler.normalize_volume(symbol, position_size)
            else:
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
        
    @classmethod
    def get_instance(cls):
        """Return the singleton instance, creating it if necessary."""
        global _risk_manager_instance
        if _risk_manager_instance is None:
            _risk_manager_instance = cls()
        return _risk_manager_instance
        
    def get_account_balance(self) -> float:
        """Return the current account balance from MT5, or 0.0 if unavailable."""
        account_info = self._get_account_info()
        if not account_info:
            return 0.0
        return account_info.get('balance', 0.0)
        
    def validate_and_size_trade(self, trade_dict: dict, strategy_id: Optional[str] = None) -> dict:
        """Validate and size a trade proposal according to all risk rules.

        Args:
            trade_dict (dict): Proposed trade (symbol, entry, stop_loss, take_profit, direction, etc.)
            strategy_id (str, optional): Strategy identifier for per-strategy risk overrides
        Returns:
            dict: {valid: bool, reason: str, adjusted_trade: dict}
        """
        # Ensure strategy_id is always a string and never None
        if strategy_id is None:
            strategy_id = ''
        else:
            strategy_id = str(strategy_id)
        # Copy trade_dict to avoid mutating input
        trade = dict(trade_dict)
        symbol = str(trade.get('symbol') or '')
        entry = trade.get('entry_price', trade.get('entry'))
        stop = trade.get('stop_loss')
        tp = trade.get('take_profit')
        direction = trade.get('direction')
        requested_size = trade.get('size', 0) or trade.get('position_size', 0)
        account_balance = self.get_account_balance() or 10000.0
        open_trades = getattr(self, 'open_trades', [])
        # Calculate lot size
        if self.use_fixed_lot_size:
            # Use fixed lot size from config, but always align with symbol's constraints
            if self.mt5_handler:
                min_lot_size = self.mt5_handler.get_symbol_min_lot_size(symbol)
                normalized_lot = self.mt5_handler.normalize_volume(symbol, self.fixed_lot_size)
                lot_size = max(min_lot_size, normalized_lot)
                lot_size = min(lot_size, self.max_lot_size)
                logger.info(f"Using fixed lot size from config (normalized): {lot_size}")
            else:
                lot_size = min(self.fixed_lot_size, self.max_lot_size)
                logger.info(f"Using fixed lot size from config (no mt5_handler): {lot_size}")
        elif requested_size > 0:
            lot_size = min(requested_size, self.max_lot_size)
        else:
            # Ensure required args are not None and correct type
            entry_val = float(entry) if entry is not None else 0.0
            stop_val = float(stop) if stop is not None else 0.0
            symbol_val = str(symbol) if symbol is not None else ''
            lot_size = self.calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=self.max_risk_per_trade * 100,
                entry_price=entry_val,
                stop_loss_price=stop_val,
                symbol=symbol_val
            )
        trade['size'] = lot_size
        # Validate trade using existing logic
        validation = self.validate_trade(trade, account_balance, open_trades)
        if not validation.get('valid'):
            return {
                'valid': False,
                'reason': validation.get('reason', 'Trade failed risk checks'),
                'adjusted_trade': None
            }
        # If position size was adjusted, update
        if 'adjusted_position_size' in validation:
            trade['size'] = validation['adjusted_position_size']
        # Optionally, adjust SL/TP if needed (future extension)
        # Return valid trade
        return {
            'valid': True,
            'reason': 'Trade validated and sized',
            'adjusted_trade': trade
        }

    def on_trade_opened(self, trade: dict) -> None:
        """Update RiskManager state when a trade is opened."""
        # Add trade to open_trades
        self.open_trades.append(trade)
        # Update daily stats
        self.daily_stats['trade_count'] += 1
        # Estimate risk for this trade (risk = abs(entry - stop) * size)
        entry = trade.get('entry_price', trade.get('entry', 0.0)) or 0.0
        stop = trade.get('stop_loss', 0.0) or 0.0
        size = trade.get('size', trade.get('position_size', 0.0)) or 0.0
        risk_amount = abs(entry - stop) * size
        self.daily_stats['total_risk'] += risk_amount
        # Optionally update realized_pnl if trade is closed with profit/loss
        # (handled in on_trade_closed)
        # Update drawdown if needed
        self._update_drawdown()

    def on_trade_closed(self, trade: dict) -> None:
        """Update RiskManager state when a trade is closed."""
        # Remove trade from open_trades (match by symbol and entry/stop or ticket)
        ticket = trade.get('ticket')
        entry = trade.get('entry_price', trade.get('entry', 0.0)) or 0.0
        stop = trade.get('stop_loss', 0.0) or 0.0
        symbol = trade.get('symbol', "") or ""
        # Remove by ticket if present, else by symbol+entry+stop
        self.open_trades = [t for t in self.open_trades if not (
            (ticket and t.get('ticket') == ticket) or
            (t.get('symbol', "") == symbol and abs(t.get('entry_price', t.get('entry', 0.0)) - entry) < 1e-6 and abs(t.get('stop_loss', 0.0) - stop) < 1e-6)
        )]
        # Update realized PnL
        profit = trade.get('profit', 0.0) or 0.0
        self.daily_stats['realized_pnl'] += profit
        # Update drawdown if needed
        self._update_drawdown()

    def should_force_close_all(self) -> bool:
        """Return True if all trades should be force-closed due to risk (e.g., max drawdown or daily loss)."""
        # Check drawdown and daily loss
        max_drawdown = getattr(self, 'max_drawdown', 0.05)
        max_daily_loss = getattr(self, 'max_daily_loss', 0.02)
        current_drawdown = self._get_current_drawdown()
        current_daily_loss = -self.daily_stats.get('realized_pnl', 0.0)
        starting_balance = self.daily_stats.get('starting_balance', 10000.0)
        if current_drawdown >= max_drawdown or current_daily_loss >= max_daily_loss * starting_balance:
            return True
        return False

    def should_force_close_trade(self, trade: dict) -> bool:
        """Return True if a specific trade should be force-closed due to risk (e.g., per-symbol drawdown)."""
        entry = trade.get('entry_price', trade.get('entry', 0.0)) or 0.0
        stop = trade.get('stop_loss', 0.0) or 0.0
        size = trade.get('size', trade.get('position_size', 0.0)) or 0.0
        account_balance = self.get_account_balance() or 10000.0
        risk_amount = abs(entry - stop) * size
        if risk_amount > self.max_risk_per_trade * account_balance:
            return True
        return False

    def get_current_risk(self) -> dict:
        """Return current risk exposure and stats."""
        return {
            'open_trades': self.open_trades,
            'total_risk': self.daily_stats.get('total_risk', 0.0),
            'drawdown': self._get_current_drawdown(),
            'realized_pnl': self.daily_stats.get('realized_pnl', 0.0),
            'trade_count': self.daily_stats.get('trade_count', 0),
            'starting_balance': self.daily_stats.get('starting_balance', 0.0)
        }

    def get_stats(self) -> dict:
        """Return summary statistics for reporting."""
        return {
            'open_trades': len(self.open_trades),
            'total_risk': self.daily_stats.get('total_risk', 0.0),
            'drawdown': self._get_current_drawdown(),
            'realized_pnl': self.daily_stats.get('realized_pnl', 0.0),
            'trade_count': self.daily_stats.get('trade_count', 0),
            'starting_balance': self.daily_stats.get('starting_balance', 0.0),
            'max_drawdown': getattr(self, 'max_drawdown', 0.05),
            'max_daily_loss': getattr(self, 'max_daily_loss', 0.02)
        }

    def format_report(self) -> str:
        """Return a formatted risk report for notifications."""
        stats = self.get_stats()
        return (
            f"Risk Report:\n"
            f"Open Trades: {stats['open_trades']}\n"
            f"Total Risk: {stats['total_risk']:.2f}\n"
            f"Drawdown: {stats['drawdown']:.2%}\n"
            f"Realized PnL: {stats['realized_pnl']:.2f}\n"
            f"Trade Count: {stats['trade_count']}\n"
            f"Starting Balance: {stats['starting_balance']:.2f}\n"
            f"Max Drawdown: {stats['max_drawdown']:.2%}\n"
            f"Max Daily Loss: {stats['max_daily_loss']:.2%}"
        )

    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown as a fraction of starting balance."""
        starting_balance = self.daily_stats.get('starting_balance', 10000.0)
        current_balance = starting_balance + self.daily_stats.get('realized_pnl', 0.0)
        peak_balance = max(current_balance, starting_balance)
        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
        return drawdown

    def _update_drawdown(self) -> None:
        """Update drawdown stats if needed (placeholder for future expansion)."""
        # This can be expanded to track max drawdown, etc.
        return None
        