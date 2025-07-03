from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from loguru import logger
import MetaTrader5 as mt5

# Use TYPE_CHECKING for import that's only used for type hints
if TYPE_CHECKING:
    from src.mt5_handler import MT5Handler
from src.utils.market_utils import calculate_pip_value, convert_pips_to_price

# Singleton instance for global reference
_risk_manager_instance = None

# Custom Exceptions for RiskManager
class RiskManagerError(Exception):
    """Base class for exceptions in RiskManager."""
    pass

class InsufficientBalanceError(RiskManagerError):
    """Raised when account balance is insufficient for an operation."""
    pass

class InvalidRiskParameterError(RiskManagerError):
    """Raised when a risk parameter (e.g., risk percent, SL) is invalid."""
    pass

class RiskCalculationError(RiskManagerError):
    """Raised when there's an error during risk calculation (e.g., position sizing)."""
    pass

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
        from config.config import RISK_MANAGER_CONFIG, TRADING_CONFIG
        self.config = RISK_MANAGER_CONFIG
        
        # Set risk parameters from config
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01)
        self.max_concurrent_trades = self.config.get('max_concurrent_trades', 2)
        # Add minimum risk:reward ratio (configurable)
        self.min_risk_reward = self.config.get('min_risk_reward', 1.0)
        
        # Core risk parameters - use timeframe-specific values if available
        self.max_daily_loss = self.config.get('max_daily_loss', 0.02)
        self.max_drawdown = 0.05  # Default to 5% max drawdown
        
        # Position management - use timeframe-specific values if available
        # self.max_weekly_trades = self.config.get('max_weekly_trades', 16) # Removed
        self.use_fixed_lot_size = TRADING_CONFIG['use_fixed_lot_size']  # Use global trading config
        self.fixed_lot_size = TRADING_CONFIG['fixed_lot_size']
        self.max_lot_size = TRADING_CONFIG['max_lot_size']
        
       
        
        # Log the timeframe-specific parameters
        logger.debug(f"Timeframe-specific risk parameters: max_risk_per_trade={self.max_risk_per_trade}, "
                    f"max_concurrent_trades={self.max_concurrent_trades}")
        
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
    ) -> None:
        """
        Validate position sizing inputs. Raises an exception if inputs are invalid.
        
        Args:
            account_balance: Account balance
            risk_per_trade: Risk per trade as decimal (e.g., 0.01 for 1%)
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            
        Raises:
            InsufficientBalanceError: If account_balance is <= 0.0
            InvalidRiskParameterError: For other invalid inputs.
        """
        # Check if account balance is valid
        if account_balance <= 0.0:
            msg = f"Invalid account balance: {account_balance}. Cannot calculate position size."
            logger.error(msg)
            raise InsufficientBalanceError(msg)
            
        # Check if risk parameter is valid
        if not (0.0 < risk_per_trade <= 1.0): # risk_per_trade should be a decimal here
            msg = f"Invalid risk percentage: {risk_per_trade*100:.2f}%. Must be between 0% and 100%."
            logger.error(msg)
            raise InvalidRiskParameterError(msg)
            
        # Check if prices are valid
        if entry_price <= 0.0:
            msg = f"Invalid entry price: {entry_price}"
            logger.error(msg)
            raise InvalidRiskParameterError(msg)
            
        if stop_loss_price <= 0.0:
            msg = f"Invalid stop loss price: {stop_loss_price}"
            logger.error(msg)
            raise InvalidRiskParameterError(msg)
            
        # Check if entry and stop loss are the same or SL is on wrong side (basic check)
        if (entry_price > stop_loss_price and entry_price - stop_loss_price < 0.00001) or \
           (stop_loss_price > entry_price and stop_loss_price - entry_price < 0.00001) or \
           (entry_price == stop_loss_price):
            msg = f"Entry price and stop loss are too close or invalid: Entry={entry_price}, SL={stop_loss_price}"
            logger.error(msg)
            raise InvalidRiskParameterError(msg)
        
        # Ensure SL is on the correct side for a potential trade
        # This is a simplified check here; full directional check happens in validate_trade
        # if (direction == "buy" and stop_loss_price >= entry_price) or \
        #    (direction == "sell" and stop_loss_price <= entry_price):
        #     msg = f"Stop loss price {stop_loss_price} is on the wrong side of entry price {entry_price} for a {direction} trade."
        #     logger.error(msg)
        #     raise InvalidRiskParameterError(msg)
            
        logger.debug("Position sizing inputs validated successfully.")

    def validate_trade(self, trade: Dict, account_balance: float, 
                     open_trades: List[Dict], 
                     correlation_matrix: Optional[Dict] = None) -> Dict:
        """
        Validate a trade against risk parameters and current market conditions.
        Enhances validation with portfolio-level risk assessment.

        Args:
            trade: Trade details (symbol, type, entry_price, stop_loss_price, take_profit_price).
            account_balance: Current account balance.
            open_trades: List of currently open trades.
            correlation_matrix: Optional correlation matrix for advanced checks.

        Returns:
            Dict: Validation result (is_valid, reason, position_size).
        """
        logger.debug(f"Initiating trade validation for trade: {trade}")
        
        symbol = trade.get('symbol')
        if not symbol:
            logger.warning("Trade validation failed: Symbol not provided in trade details.")
            return {'is_valid': False, 'reason': "Symbol not provided", 'position_size': 0.0}

        # Basic trade parameter validation
        if 'entry_price' not in trade:
            logger.error("Trade validation failed: entry_price is missing.")
            return {'is_valid': False, 'reason': "entry_price is missing", 'position_size': 0.0}
        
        # Check for direction (either 'type' or 'direction')
        has_direction = 'type' in trade or 'direction' in trade
        if not has_direction:
            logger.error("Trade validation failed: Direction key ('type' or 'direction') is missing.")
            return {'is_valid': False, 'reason': "Direction key ('type' or 'direction') is missing", 'position_size': 0.0}

        has_stop_loss = 'stop_loss_price' in trade or 'stop_loss' in trade
        if not has_stop_loss:
            logger.error("Trade validation failed: Stop loss (stop_loss_price or stop_loss) is missing.")
            return {'is_valid': False, 'reason': "Stop loss key (stop_loss_price or stop_loss) is missing", 'position_size': 0.0}

        # --- MODIFICATION: Handle single or multiple take-profits ---
        has_take_profits = 'take_profits' in trade and isinstance(trade.get('take_profits'), list)
        has_single_take_profit = 'take_profit_price' in trade or 'take_profit' in trade
        
        if not has_take_profits and not has_single_take_profit:
            logger.error("Trade validation failed: Take profit key ('take_profits', 'take_profit_price', or 'take_profit') is missing.")
            return {'is_valid': False, 'reason': "Take profit key is missing", 'position_size': 0.0}

        entry_val = trade.get('entry_price') 
        stop_val = trade.get('stop_loss_price', trade.get('stop_loss'))
        
        # --- Multi-TP Handling ---
        tp_val = None
        if has_take_profits:
            tp_list = trade.get('take_profits', [])
            if not tp_list:
                logger.error("Trade validation failed: 'take_profits' list is present but empty.")
                return {'is_valid': False, 'reason': "'take_profits' list is empty", 'position_size': 0.0}
            tp_val = tp_list[0] # Use the first TP for validation
            logger.debug(f"Validating with first TP from list: {tp_val}")
        else:
            tp_val = trade.get('take_profit_price', trade.get('take_profit'))

        if entry_val is None: # Should have been caught by 'entry_price' in trade check, but for safety
            logger.error("Trade validation failed: entry_price value is None.")
            return {'is_valid': False, 'reason': "entry_price value is None", 'position_size': 0.0}
        if stop_val is None:
            logger.error("Trade validation failed: Stop loss value is None after checking both keys.")
            return {'is_valid': False, 'reason': "Stop loss value is None", 'position_size': 0.0}
        if tp_val is None:
            logger.error("Trade validation failed: Take profit value is None after checking both keys.")
            return {'is_valid': False, 'reason': "Take profit value is None", 'position_size': 0.0}
        
        # At this point, entry_val, stop_val, tp_val are confirmed not None.
        try:
            entry = float(entry_val)
            stop = float(stop_val)
            tp = float(tp_val)
        except (ValueError, TypeError) as e:
            logger.error(f"Trade validation failed: entry/stop/tp could not be converted to float. Error: {e}. Values: E={entry_val}, S={stop_val}, TP={tp_val}")
            return {'is_valid': False, 'reason': f"Invalid E/S/TP format: {e}", 'position_size': 0.0}

        # Get direction, preferring 'type' then 'direction'
        direction = trade.get('type', trade.get('direction'))
        if direction is None: # Should be caught by has_direction check, but for safety
            logger.error("Trade validation failed: Direction value is None after checking both keys.")
            return {'is_valid': False, 'reason': "Direction value is None", 'position_size': 0.0}
        
        requested_size = trade.get('position_size', 0) # From strategy or signal

        # --- Initial Position Size Determination (before portfolio check) ---
        position_size: float = 0.0
        min_lot_from_symbol: float = 0.01 # Default minimum
        contract_size_from_symbol: float = 1.0 # Default contract size

        if self.mt5_handler:
            symbol_info_initial = self.mt5_handler.get_symbol_info(symbol)
            if symbol_info_initial: # Check if it's not None
                try:
                    min_lot_from_symbol = getattr(symbol_info_initial, 'volume_min', 0.01)
                    contract_size_from_symbol = getattr(symbol_info_initial, 'trade_contract_size', 1.0)
                except AttributeError as e:
                    logger.warning(f"Attribute error accessing symbol info for {symbol}: {e}. Using defaults.")
            else:
                logger.warning(f"Could not retrieve symbol info for {symbol} during initial size determination. Using defaults.")
        else:
            logger.warning("MT5 handler not available for initial size determination. Using defaults.")

        if self.use_fixed_lot_size:
            position_size = self.fixed_lot_size
            if self.mt5_handler and symbol_info_initial: # Check if symbol_info_initial was fetched
                if self.fixed_lot_size < min_lot_from_symbol:
                    position_size = min_lot_from_symbol
                    logger.info(f"Fixed lot size {self.fixed_lot_size} is below min {min_lot_from_symbol} for {symbol}, using min.")
                else:
                    position_size = self.mt5_handler.normalize_volume(symbol, self.fixed_lot_size)
            position_size = min(position_size, self.max_lot_size) # Apply max lot cap
            logger.info(f"Using fixed lot size for {symbol}: {position_size}")
        elif requested_size > 0:
            position_size = requested_size
            if self.mt5_handler and symbol_info_initial:
                position_size = max(requested_size, min_lot_from_symbol)
                position_size = self.mt5_handler.normalize_volume(symbol, position_size)
            position_size = min(position_size, self.max_lot_size) # Apply max lot cap
            logger.info(f"Using requested position_size (adjusted): {position_size}")
        else:
            try:
                # risk_per_trade in self.max_risk_per_trade is already a decimal e.g. 0.01 for 1%
                # calculate_position_size expects it as a raw percentage e.g. 1.0 for 1%
                risk_percentage_for_calc = self.max_risk_per_trade * 100.0
                
                position_size = self.calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=risk_percentage_for_calc, 
                    entry_price=entry,
                    stop_loss_price=stop,
                    symbol=symbol
                )
                if self.mt5_handler and symbol_info_initial:
                     # Ensure calculated size respects symbol's min and step
                    position_size = max(position_size, min_lot_from_symbol)
                    position_size = self.mt5_handler.normalize_volume(symbol, position_size)
                position_size = min(position_size, self.max_lot_size) # Apply max lot cap
                logger.info(f"Calculated position size based on risk: {position_size}")
            except (InvalidRiskParameterError, RiskCalculationError) as e: # Catch specific errors
                logger.error(f"Trade validation failed due to error in position sizing for {symbol}: {str(e)}")
                return {'is_valid': False, 'reason': f"Position sizing error: {str(e)}", 'position_size': 0.0}
            except Exception as e: # Catch any other unexpected error during calculation
                logger.error(f"Unexpected error calculating position size for {symbol}: {str(e)}. Trade invalid.")
                return {'is_valid': False, 'reason': f"Unexpected position sizing error: {str(e)}", 'position_size': 0.0}
        
        if position_size <= 0:
            logger.warning(f"Initial position size for {symbol} is {position_size}. Trade invalid.")
            return {'is_valid': False, 'reason': f"Initial position size is not positive: {position_size}", 'position_size': 0.0}

        # --- Portfolio Level Risk Assessment ---
        logger.info(f"Performing portfolio level risk assessment for symbol: {symbol}")
        portfolio_risk_analysis = self.calculate_portfolio_risk_limits(tickers_to_analyze=[symbol])
        
        if "error" in portfolio_risk_analysis:
            logger.warning(f"Portfolio risk calculation failed: {portfolio_risk_analysis.get('error')}")
            return {'is_valid': False, 'reason': f"Portfolio risk calculation error: {portfolio_risk_analysis.get('error')}", 'position_size': 0.0}

        ticker_portfolio_analysis = portfolio_risk_analysis.get(symbol)

        if not ticker_portfolio_analysis:
            logger.warning(f"No portfolio analysis data found for symbol {symbol}.")
            return {'is_valid': False, 'reason': f"No portfolio analysis for {symbol}", 'position_size': 0.0}
        
        remaining_cash_limit_from_portfolio = ticker_portfolio_analysis.get('remaining_position_limit', 0.0)
        current_price_from_portfolio_calc = ticker_portfolio_analysis.get('current_price', 0.0)

        if current_price_from_portfolio_calc == 0.0:
            logger.warning(f"Trade validation failed for {symbol}: Current price from portfolio calculation is zero.")
            return {'is_valid': False, 'reason': "Current price from portfolio calculation is zero", 'position_size': 0.0}
            
        logger.info(f"Portfolio analysis for {symbol}: Remaining Cash Limit: {remaining_cash_limit_from_portfolio:.2f}, Current Price: {current_price_from_portfolio_calc:.5f}")
        
        # --- Apply Portfolio Limit to Position Size ---
        if self.use_fixed_lot_size:
            # For fixed lot size, position_size is already determined. We just check if its value exceeds the limit.
            # Value of the fixed lot trade = position_size (lots) * contract_size * price_per_unit
            value_of_fixed_trade = position_size * contract_size_from_symbol * current_price_from_portfolio_calc
            if value_of_fixed_trade > remaining_cash_limit_from_portfolio:
                logger.warning(f"Trade validation failed for {symbol}: Fixed lot size ({position_size}) value ({value_of_fixed_trade:.2f}) exceeds remaining portfolio cash limit ({remaining_cash_limit_from_portfolio:.2f}).")
                return {'is_valid': False, 'reason': "Fixed lot size value exceeds portfolio cash limit", 'position_size': 0.0}
        else:
            # For dynamic lot size, position_size is already calculated. Adjust if it exceeds portfolio limit.
            value_per_lot = contract_size_from_symbol * current_price_from_portfolio_calc
            if value_per_lot <= 0: # Avoid division by zero or invalid calculations
                 logger.warning(f"Trade validation failed for {symbol}: Value per lot ({value_per_lot:.2f}) is not positive, cannot assess against portfolio limit.")
                 return {'is_valid': False, 'reason': "Value per lot is not positive for portfolio limit check", 'position_size': 0.0}

            max_lots_from_portfolio_limit = remaining_cash_limit_from_portfolio / value_per_lot
            
            if position_size > max_lots_from_portfolio_limit:
                logger.warning(f"Calculated position size ({position_size} lots) for {symbol} exceeds max lots based on portfolio limit ({max_lots_from_portfolio_limit:.4f}). Adjusting.")
                position_size = max_lots_from_portfolio_limit
                # Ensure it doesn't go below minimum lot size or become zero, and normalize
                if self.mt5_handler and symbol_info_initial: # symbol_info_initial should be available
                    position_size = max(min_lot_from_symbol, position_size)
                    position_size = self.mt5_handler.normalize_volume(symbol, position_size)
                else: # Basic rounding if no handler/info
                    position_size = round(max(0.01, position_size), 2) # Ensure at least 0.01 and round
                
                if position_size < min_lot_from_symbol: # Final check against actual min_lot
                    logger.warning(f"Trade validation failed for {symbol}: Adjusted position size ({position_size}) is below minimum lot size ({min_lot_from_symbol}) after portfolio limit.")
                    return {'is_valid': False, 'reason': f"Adjusted size ({position_size}) below min_lot ({min_lot_from_symbol}) due to portfolio limit", 'position_size': 0.0}
                logger.info(f"Position size for {symbol} adjusted to {position_size} lots due to portfolio cash limit.")
        # --- End Portfolio Level Logic ---

        
        if direction is None: # Should have been caught earlier but double check
            logger.error("Direction is None before R:R calculation, this indicates a logic flaw.")
            return {'is_valid': False, 'reason': "Internal error: Direction became None", 'position_size': 0.0}
            
        direction_lower = str(direction).lower() # Correctly use the 'direction' variable

        if direction_lower == "buy":
            risk_val = entry - stop           # should be > 0
            reward_val = tp - entry           # should be > 0
        elif direction_lower == "sell":
            risk_val = stop - entry          # should be > 0
            reward_val = entry - tp          # should be > 0
        else:
            # Unknown direction – treat as invalid
            return {
                'is_valid': False,
                'reason': f"Unknown trade direction '{direction}'"
            }

        # Validate SL/TP placement
        if risk_val <= 0 or reward_val <= 0:
            logger.warning(f"Trade validation failed for {symbol}: Stop-loss or take-profit is incorrectly placed. Risk: {risk_val}, Reward: {reward_val}")
            return {
                'is_valid': False,
                'reason': 'Stop-loss and/or take-profit are on the wrong side of the entry price for the trade direction'
            }
        # Additional: TP must be above entry for buy, below for sell
        if direction_lower == "buy" and tp <= entry:
            return {
                'is_valid': False,
                'reason': 'Take profit must be above entry price for buy orders'
            }
        elif direction_lower == "sell" and tp >= entry:
            return {
                'is_valid': False,
                'reason': 'Take profit must be below entry price for sell orders'
            }

        # Enforce minimum R:R
        if (reward_val / risk_val) < self.min_risk_reward:
            logger.warning(f"Trade validation failed for {symbol}: Risk:Reward ratio ({reward_val / risk_val:.2f}) is below minimum ({self.min_risk_reward:.2f}).")
            return {
                'is_valid': False,
                'reason': f"Risk:Reward ratio too low ({reward_val / risk_val:.2f}), must be at least {self.min_risk_reward:.2f}"
            }
       
        if stop != 0 and entry != 0: # Ensure SL and entry are valid for risk calculation
            # Calculate risk amount based on the potentially adjusted position_size
            risk_amount = 0.0
            if self.mt5_handler:
                symbol_info_for_risk_calc = self.mt5_handler.get_symbol_info(symbol)
                point_value = 0.00001 # Default point value
                
                if symbol_info_for_risk_calc: # Check if it's not None
                    try:
                        point_value = getattr(symbol_info_for_risk_calc, 'point', 0.00001)
                    except AttributeError as e:
                        logger.warning(f"Attribute error accessing point from symbol_info_for_risk_calc for {symbol}: {e}. Type: {type(symbol_info_for_risk_calc)}. Using default.")
                else:
                    logger.warning(f"Symbol info for risk calculation not found for {symbol}. Using default point value.")

                if point_value == 0: # Avoid division by zero if point is somehow 0
                    logger.warning(f"Point value for {symbol} is zero. Cannot accurately calculate pips at risk. Defaulting risk_amount to a high indicative value or skipping.")
                    # Fallback or error, as pip calculation would be problematic
                    # For now, let's make risk_amount high to likely fail the trade if this occurs
                    risk_amount = account_balance # This will likely fail the risk_percentage check
                else:
                    pips_at_risk = abs(entry - stop) / point_value
                   
                    risk_amount = abs(entry - stop) * position_size * contract_size_from_symbol

            else: # Fallback if no mt5_handler
                logger.warning("MT5 handler not available for risk amount calculation. Using approximate risk.")
                # Approximate risk based on price change and size, assuming standard contract
                risk_amount = abs(entry - stop) * position_size * 1 # Assuming contract size of 1 if no info

            if account_balance > 0: # Avoid division by zero if account_balance is somehow zero
                risk_percentage = risk_amount / account_balance
                logger.info(f"Risk per trade check for {symbol}: Size={position_size}, Entry={entry}, SL={stop}, Risk Amount={risk_amount:.2f}, Balance={account_balance:.2f}, Risk Percentage={risk_percentage:.4f}")

                if risk_percentage > self.max_risk_per_trade:
                    logger.warning(f"Trade validation failed for {symbol}: Risk per trade ({risk_percentage*100:.2f}%) exceeds max ({self.max_risk_per_trade*100}%). Position Size: {position_size}")
                    # Optionally, you could try to readjust size here again, but it might conflict with portfolio limits.
                    # For now, if portfolio limit was met, but this rule is now breached by that adjusted size, it's a fail.
                    return {
                        'is_valid': False,
                        'reason': f"Risk per trade ({risk_percentage*100:.2f}%) exceeds max ({self.max_risk_per_trade*100}%) after all adjustments",
                        'position_size': 0.0
                    }
            else:
                logger.warning(f"Account balance is zero or negative for {symbol}. Cannot calculate risk percentage.")
                return {'is_valid': False, 'reason': "Account balance zero or negative for risk calculation", 'position_size': 0.0}

        # Final check on max concurrent trades (can be done earlier too)
        if len(open_trades) >= self.max_concurrent_trades:
             logger.warning(f"Trade validation failed for {symbol}: Max concurrent trades ({self.max_concurrent_trades}) would be exceeded.")
             return {
                 'is_valid': False,
                 'reason': f"Max concurrent trades ({self.max_concurrent_trades}) reached",
                 'position_size': 0.0
             }

        logger.info(f"Trade validation successful for {symbol} with position size: {position_size}")
        return {
            'is_valid': True,
            'reason': "Trade meets all risk management criteria",
            'position_size': position_size
        }

    def initialize(self, timeframe=None):
        """
        Initialize or reinitialize the RiskManager.
        Currently, this method logs the timeframe but applies the global RISK_MANAGER_CONFIG.
        Timeframe-specific configurations are not loaded differently at this stage.
        
        Args:
            timeframe: Optional timeframe to log, does not change loaded config parameters.
        """
        if timeframe:
            # Update the current timeframe if provided
            # self.timeframe = timeframe # Consider if self.timeframe is used elsewhere
            logger.info(f"RiskManager initialize called with timeframe: {timeframe}. Global risk config will be applied.")
            
            # Load timeframe-specific parameters (currently uses global config)
            try:
                from config.config import RISK_MANAGER_CONFIG # Changed import
                timeframe_config = RISK_MANAGER_CONFIG       # Use global config
                
                # Update core risk parameters with (currently global) timeframe-specific values
                self.max_risk_per_trade = timeframe_config.get('max_risk_per_trade', self.max_risk_per_trade)
                self.max_daily_loss = timeframe_config.get('max_daily_loss', self.max_daily_loss)
                
                # Update position management with timeframe-specific values
                self.max_concurrent_trades = timeframe_config.get('max_concurrent_trades', self.max_concurrent_trades)
                
                logger.info(f"RiskManager updated with {timeframe} timeframe parameters")
            except Exception as e:
                logger.error(f"Error loading timeframe-specific parameters: {e}")
        
        logger.info("No custom config provided for RiskManager, using current settings")
        return
       
        logger.info("RiskManager re-initialized (using global RISK_MANAGER_CONFIG).")

        
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float, # Raw percentage, e.g., 1.0 for 1%
        entry_price: float,
        stop_loss_price: float,
        symbol: str
    ) -> float:
        """
        Calculate position size based on account balance, risk percentage, and stop loss distance.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage (e.g., 1.0 for 1%)
            entry_price: Entry price of the trade
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            
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
                    
                    # If the fixed lot size is less than the symbol minimum,
                    # use the symbol's minimum lot size directly
                    if position_size < min_lot_size:
                        position_size = min_lot_size
                        logger.info(f"Fixed lot size {self.fixed_lot_size} is below minimum lot size {min_lot_size} for {symbol}, using minimum")
                    else:
                        # Normalize volume according to symbol's volume_step
                        position_size = self.mt5_handler.normalize_volume(symbol, position_size)
                        logger.info(f"Normalized fixed lot size to {position_size} for {symbol}")
                
                logger.info(f"Using fixed lot size of {position_size} for {symbol}")
                return position_size
                
            # If not using fixed lot size, calculate based on risk
            # Validate inputs first
            self._validate_position_inputs(account_balance, risk_per_trade / 100.0, entry_price, stop_loss_price)
            # If _validate_position_inputs passes, we can proceed. It raises exceptions on failure.
                
            # Calculate risk amount
            risk_amount = account_balance * (risk_per_trade / 100.0) # Convert risk_per_trade percentage to decimal
            
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss_price)
            if stop_distance <= 0: # This should ideally be caught by _validate_position_inputs
                msg = f"Invalid stop distance: {stop_distance} for {symbol}. Entry: {entry_price}, SL: {stop_loss_price}"
                logger.error(msg)
                raise RiskCalculationError(msg) # Raise specific error
                
            # Calculate risk per pip
            risk_per_pip = risk_amount / stop_distance
            
            # Use symbol's point value instead of hardcoded 0.0001
            point_value = 0.0001
            if self.mt5_handler and hasattr(self.mt5_handler, 'get_point_value'):
                try:
                    point_value = self.mt5_handler.get_point_value(symbol)
                except Exception as e:
                    logger.warning(f"Could not fetch point value for {symbol}: {e}. Using default 0.0001.")
            position_size = risk_per_pip * point_value  # FIXED: use symbol's point value
            
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
            
            logger.info(f"Calculated position size: {position_size} lots for {symbol} based on risk")
            return position_size
            
        except (InvalidRiskParameterError, InsufficientBalanceError) as e: # Catch specific validation errors
            logger.error(f"Validation error in calculate_position_size for {symbol}: {str(e)}")
            raise # Re-raise to be caught by validate_trade or other callers
        except RiskCalculationError as e: # Catch specific calculation errors
            logger.error(f"Calculation error in calculate_position_size for {symbol}: {str(e)}")
            raise # Re-raise
        except Exception as e:
            # Catch any other unexpected error and wrap it
            logger.error(f"Unexpected error calculating position size for {symbol}: {str(e)}")
            raise RiskCalculationError(f"Unexpected error during position sizing for {symbol}: {str(e)}")
        
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
        
        trade_input = dict(trade_dict) # Operate on a copy
        
        account_balance = self.get_account_balance()
        if account_balance <= 0:
            logger.warning(f"Account balance is {account_balance}. Cannot validate or size trade.")
            return {
                'is_valid': False, 
                'reason': f"Invalid account balance: {account_balance}",
                'final_trade_params': None
            }

        open_trades = getattr(self, 'open_trades', [])
        if not open_trades and self.mt5_handler:
            logger.debug("RiskManager.open_trades is empty, fetching from MT5Handler for validate_and_size_trade.")
            fetched_positions = self.mt5_handler.get_open_positions()
            open_trades = fetched_positions if fetched_positions is not None else []
        
        logger.debug(f"Calling comprehensive validate_trade with trade_input: {trade_input}, balance: {account_balance}")
        validation_result = self.validate_trade(trade_input, account_balance, open_trades)

        if not validation_result.get('is_valid'):
            logger.warning(f"Trade validation failed by validate_trade: {validation_result.get('reason')}")
            return {
                'is_valid': False,
                'reason': validation_result.get('reason', 'Trade failed comprehensive risk checks'),
                'final_trade_params': None
            }

        final_position_size = validation_result.get('position_size')
        if final_position_size is None or final_position_size <= 0:
            logger.error(f"Validation succeeded but returned invalid position_size: {final_position_size}. Trade considered invalid.")
            return {
                'is_valid': False,
                'reason': f"Validation succeeded but returned invalid final position size: {final_position_size}",
                'final_trade_params': None
            }

        final_trade_params = dict(trade_input)
        final_trade_params['position_size'] = final_position_size
        final_trade_params['size'] = final_position_size 

        logger.info(f"Trade validated and sized successfully. Final parameters: {final_trade_params}")
        return {
            'is_valid': True,
            'reason': validation_result.get('reason', 'Trade validated and sized'),
            'final_trade_params': final_trade_params
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

    def calculate_portfolio_risk_limits(self, tickers_to_analyze: List[str]) -> Dict[str, Any]:
        
        logger.info(f"Calculating portfolio risk limits. Initial tickers to analyze: {tickers_to_analyze}")
        risk_analysis: Dict[str, Any] = {}
        current_prices: Dict[str, float] = {}

        if not self.mt5_handler:
            logger.error("MT5Handler not available in RiskManager. Cannot proceed with portfolio risk calculation.")
            for ticker in tickers_to_analyze:
                risk_analysis[ticker] = {
                    "current_price": 0.0,
                    "remaining_position_limit": 0.0,
                    "reasoning": {"error": "MT5Handler not available"},
                }
            return risk_analysis

        # Step 1: Get Account Info (Cash)
        account_info = self.mt5_handler.get_account_info()
        cash = float(account_info.get("balance", 0.0))
        logger.debug(f"Fetched cash (account balance): {cash}")

        # Step 2: Get Open Positions from MT5
        mt5_positions = self.mt5_handler.get_open_positions() # Returns list of dicts
        logger.debug(f"Fetched {len(mt5_positions)} open positions from MT5.")

        # Step 3: Determine all unique tickers involved (to analyze + existing positions)
        all_relevant_tickers = set(tickers_to_analyze)
        for pos in mt5_positions:
            if pos.get("symbol"): 
                all_relevant_tickers.add(pos["symbol"])
        logger.debug(f"All relevant tickers for price fetching: {list(all_relevant_tickers)}")

        # Step 4: Fetch current prices for ALL relevant tickers
        for ticker in all_relevant_tickers:
            price = 0.0
            try:
                tick_data = self.mt5_handler.get_last_tick(ticker)
                if tick_data and isinstance(tick_data, dict):
                    fetched_price = tick_data.get('ask') # Default to ask for new potential trades
                    
                    # Find positions matching the current ticker
                    matching_positions = []
                    for p in mt5_positions: # mt5_positions is List[Dict[str, Any]]
                        pos_symbol = p.get("symbol") # pos_symbol could be None or Not Str
                        # Ensure pos_symbol is a string and matches the ticker before appending
                        if isinstance(pos_symbol, str) and pos_symbol == ticker:
                            matching_positions.append(p)

                    if matching_positions: # Check if the list is not empty
                        # If there are open positions for this ticker, use their current price for valuation
                        # Taking the first one if multiple (though usually one aggregate position in MT5)
                        # .get on matching_positions[0] (a Dict) is fine.
                        fetched_price = matching_positions[0].get('current_price', fetched_price)
                    
                    if fetched_price is not None and float(fetched_price) > 0:
                        price = float(fetched_price)
                        current_prices[ticker] = price
                        logger.debug(f"Fetched/confirmed current price for {ticker}: {price}")
                    else:
                        logger.warning(f"Could not get a valid price for {ticker}. Defaulting to 0.0. Tick: {tick_data}")
                        current_prices[ticker] = 0.0
                else:
                    logger.warning(f"No tick data found for {ticker}. Defaulting price to 0.0.")
                    current_prices[ticker] = 0.0
            except Exception as e:
                logger.error(f"Exception fetching price for {ticker}: {e}")
                current_prices[ticker] = 0.0
        
        # Initialize risk_analysis structure for all tickers_to_analyze
        for ticker in tickers_to_analyze:
            risk_analysis[ticker] = {
                "current_price": current_prices.get(ticker, 0.0),
                "remaining_position_limit": 0.0, # To be calculated later
                "reasoning": {} # To be populated later
            }
            if current_prices.get(ticker, 0.0) == 0.0:
                 risk_analysis[ticker]["reasoning"]["error"] = "Failed to fetch valid price for limit calculation"

        # Step 5: Transform MT5 positions and calculate their market value
        # Structure: {'TICKER': {'long_volume': float, 'short_volume': float, 'long_value': float, 'short_value': float, 'net_value': float}}
        transformed_positions: Dict[str, Dict[str, float]] = {}
        total_long_value = 0.0
        total_short_value = 0.0

        for pos in mt5_positions:
            symbol_from_pos = pos.get("symbol") 
            
            if not isinstance(symbol_from_pos, str): 
                logger.warning(f"Skipping position transformation for position with ticket {pos.get('ticket', 'N/A')} because its symbol is not a valid string or is missing: '{symbol_from_pos}'")
                continue

            # At this point, symbol_from_pos is confirmed to be a str.
            volume = float(pos.get("volume", 0.0))
            pos_type = pos.get("type") 
            # Explicitly use the validated string variable for the key.
            price_at_valuation = current_prices.get(symbol_from_pos, 0.0) 
            
            if price_at_valuation == 0.0:
                logger.warning(f"Skipping position transformation for position with ticket {pos.get('ticket', 'N/A')} (Symbol: {symbol_from_pos}) due to zero/missing price_at_valuation.")
                continue

            # Ensure all subsequent uses of the position's symbol use symbol_from_pos
            if symbol_from_pos not in transformed_positions:
                transformed_positions[symbol_from_pos] = {'long_volume': 0.0, 'short_volume': 0.0, 'long_value': 0.0, 'short_value': 0.0, 'net_value': 0.0}
            
            market_value = volume * price_at_valuation

            if pos_type == 0: # BUY (Long)
                transformed_positions[symbol_from_pos]['long_volume'] += volume
                transformed_positions[symbol_from_pos]['long_value'] += market_value
                total_long_value += market_value
            elif pos_type == 1: # SELL (Short)
                transformed_positions[symbol_from_pos]['short_volume'] += volume
                transformed_positions[symbol_from_pos]['short_value'] += market_value # Market value of shares owed
                total_short_value += market_value
        
        for symbol_data_key in list(transformed_positions.keys()): # Iterate over keys to allow modification
            # Ensure symbol_data_key is used if it was the one validated
            symbol_data = transformed_positions[symbol_data_key]
            symbol_data['net_value'] = symbol_data['long_value'] - symbol_data['short_value']

        logger.debug(f"Transformed positions: {transformed_positions}")
        logger.debug(f"Total long value: {total_long_value}, Total short value: {total_short_value}")

        # Step 6: Calculate Total Portfolio Value
        # total_portfolio_value = cash + sum of (long_qty * price) - sum of (short_qty * price)
        total_portfolio_value = cash + total_long_value - total_short_value
        logger.info(f"Calculated Total Portfolio Value: {total_portfolio_value:.2f} (Cash: {cash:.2f} + Longs: {total_long_value:.2f} - Shorts: {total_short_value:.2f})")

        # Step 7: Calculate risk limits for each ticker in tickers_to_analyze
        for ticker in tickers_to_analyze:
            current_price_for_ticker = current_prices.get(ticker, 0.0)
            analysis_entry = risk_analysis[ticker] # Get the pre-initialized entry

            analysis_entry["reasoning"]["portfolio_cash"] = float(cash)
            analysis_entry["reasoning"]["total_portfolio_value"] = float(total_portfolio_value)
            
            if current_price_for_ticker == 0.0:
                logger.warning(f"Cannot calculate limits for {ticker}; current price is 0 or unavailable.")
                analysis_entry["remaining_position_limit"] = 0.0
                if "error" not in analysis_entry["reasoning"]:
                    analysis_entry["reasoning"]["error"] = "Missing price for limit calculation"
                continue

            # Current market value of this specific ticker's position (if any)
            ticker_position_data = transformed_positions.get(ticker, {'long_value': 0.0, 'short_value': 0.0})
            # Absolute exposure for this ticker
            current_market_value_of_ticker_position = abs(ticker_position_data['long_value'] - ticker_position_data['short_value'])
            analysis_entry["reasoning"]["current_market_value_of_ticker_position"] = float(current_market_value_of_ticker_position)

            # Position limit (e.g., 20% of total portfolio value per ticker)
            # This percentage should ideally be configurable.
            max_single_ticker_exposure_percentage = 0.20 # As per original agent snippet
            cash_limit_per_ticker = total_portfolio_value * max_single_ticker_exposure_percentage
            analysis_entry["reasoning"]["cash_limit_per_ticker_raw_ (20%)"] = float(cash_limit_per_ticker)

            # Remaining cash limit for *new* trades on this ticker
            remaining_cash_for_new_trades_on_ticker = cash_limit_per_ticker - current_market_value_of_ticker_position
            analysis_entry["reasoning"]["remaining_cash_for_new_trades_on_ticker_(limit-current_exposure)"] = float(remaining_cash_for_new_trades_on_ticker)
            
            # Ensure this remaining limit does not exceed available cash in the account
            # Also, ensure it's not negative (meaning already over-exposed or limit is negative due to low portfolio value)
            final_remaining_cash_allocation = max(0.0, min(remaining_cash_for_new_trades_on_ticker, cash))
            analysis_entry["remaining_position_limit"] = float(final_remaining_cash_allocation)
            analysis_entry["reasoning"]["final_remaining_cash_allocation_(capped_by_available_cash_and_non_negative)"] = float(final_remaining_cash_allocation)

            logger.info(f"Risk limits for {ticker}: Price={current_price_for_ticker:.5f}, TotalPortfolioVal={total_portfolio_value:.2f}, LimitPerTicker={cash_limit_per_ticker:.2f}, CurrentExposure={current_market_value_of_ticker_position:.2f}, RemainingCashAllocation={final_remaining_cash_allocation:.2f}")

        logger.info(f"Completed portfolio risk limit calculation. Full Analysis: {risk_analysis}")
        return risk_analysis
        