from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from loguru import logger
# import MetaTrader5 as mt5 # Removed: No direct MT5 import needed for backtest-focused RM

# Use TYPE_CHECKING for import that's only used for type hints
if TYPE_CHECKING:
    from src.mt5_handler import MT5Handler # Keep for type hinting if live mode is ever re-enabled in this file
    # If MT5Handler is completely removed, this can go too. For now, conditional.
    import MetaTrader5 as mt5 # Keep for type hinting if live mode code refers to mt5 constants

from src.utils.market_utils import calculate_pip_value, convert_pips_to_price

# Singleton instance for global reference - Primarily for live mode.
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

    def __init__(self,
                 # mt5_handler: Optional['MT5Handler'] = None, # Removed for backtest focus
                 backtest_mode: bool = True, # Defaulting to True
                 backtest_initial_balance: float = 10000.0,
                 backtest_symbol_info: Optional[Dict[str, Any]] = None,
                 config_override: Optional[Dict[str, Any]] = None): # For passing RISK_MANAGER_CONFIG, TRADING_CONFIG
        """
        Initialize the risk manager. In backtest_mode, it operates without live MT5 calls.
        
        Args:
            backtest_mode: If True, operates in backtesting mode. Currently always True.
            backtest_initial_balance: Initial balance for backtesting.
            backtest_symbol_info: Dict providing symbol details for backtesting 
                                  (e.g., { "XAUUSD": {"pip_size": 0.01, "contract_size": 100, ...} }).
            config_override: Dict to override RISK_MANAGER_CONFIG and TRADING_CONFIG for testing.
        """
        global _risk_manager_instance

        # Simplified singleton logic: only relevant if a live mode is re-introduced.
        # For backtesting, always create a new instance.
        if not backtest_mode: # This block would be for a future live mode
            if _risk_manager_instance is not None:
                logger.info("Using existing RiskManager instance for live mode.")
                self.__dict__ = _risk_manager_instance.__dict__
                self.backtest_mode = backtest_mode # Ensure this instance knows its mode
                return
            _risk_manager_instance = self
            # self.mt5_handler = mt5_handler # Or get instance
            # if self.mt5_handler is None:
            #     from src.mt5_handler import MT5Handler
            #     try:
            #         self.mt5_handler = MT5Handler.get_instance()
            #     except Exception as e:
            #         logger.error(f"Failed to get MT5Handler instance: {e}")
            #         self.mt5_handler = None
            # self.mt5 = self.mt5_handler # This would be the MT5Handler instance, not the library
            logger.warning("RiskManager initialized in non-backtest mode. This mode is not fully supported in backtest_risk_manager.py")

        else: # backtest_mode is True
            logger.info("RiskManager initialized in backtest mode.")
            # _risk_manager_instance = None # Do not assign to singleton in backtest mode.

        self.backtest_mode = True # Hardcoded to True for this file's purpose
        self.backtest_initial_balance = backtest_initial_balance
        self.backtest_symbol_info = backtest_symbol_info if backtest_symbol_info else {}
        
        # No mt5_handler or mt5 library direct usage in backtest mode
        self.mt5_handler: Optional['MT5Handler'] = None 
        # self.mt5 = None # No direct mt5 library usage

        if config_override:
            risk_manager_config_dict = config_override.get('RISK_MANAGER_CONFIG', {})
            trading_config_dict = config_override.get('TRADING_CONFIG', {})
        else:
            from config.config import RISK_MANAGER_CONFIG, TRADING_CONFIG
            risk_manager_config_dict = RISK_MANAGER_CONFIG
            trading_config_dict = TRADING_CONFIG
        
        self.config = risk_manager_config_dict # Store the risk manager specific part
        
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01)
        self.max_concurrent_trades = self.config.get('max_concurrent_trades', 2)
        self.min_risk_reward = self.config.get('min_risk_reward', 1.0)
        
        self.max_daily_loss = self.config.get('max_daily_loss', 0.02)
        self.max_drawdown = self.config.get('max_drawdown', 0.05)
        
        self.use_fixed_lot_size = trading_config_dict.get('use_fixed_lot_size', False)
        self.fixed_lot_size = trading_config_dict.get('fixed_lot_size', 0.01)
        self.max_lot_size = trading_config_dict.get('max_lot_size', 10.0) # Crucial for backtesting
        
        logger.debug(f"Risk parameters (backtest mode): max_risk_per_trade={self.max_risk_per_trade}, "
                    f"max_concurrent_trades={self.max_concurrent_trades}, "
                    f"max_daily_loss={self.max_daily_loss}, max_drawdown={self.max_drawdown}, "
                    f"max_lot_size={self.max_lot_size}")
        
        self.daily_stats = {
            'total_risk': 0.0,
            'realized_pnl': 0.0,
            'trade_count': 0,
            'starting_balance': 0.0,
            'last_reset': datetime.now(UTC).date() # Can be simplified for backtesting if daily resets aren't simulated
        }

        self.open_trades: List[Dict[str, Any]] = []
        self._update_starting_balance()

    
    def _update_starting_balance(self) -> None:
        """Update the starting balance. In backtest mode, uses backtest_initial_balance."""
        # if self.backtest_mode: # This condition is always true now
        self.daily_stats['starting_balance'] = self.backtest_initial_balance
        logger.info(f"Backtest mode: Starting balance set to {self.backtest_initial_balance}")
        # else:
            # Live mode logic would go here, e.g.:
            # try:
            #     account_info = self._get_account_info()
            #     if account_info and 'balance' in account_info:
            #         self.daily_stats['starting_balance'] = account_info['balance']
            #     elif not account_info:
            #         logger.warning("Could not retrieve account info to update starting balance.")
            # except Exception as e:
            #     logger.error(f"Error updating starting balance: {str(e)}")
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get account information. In backtest mode, returns backtest defaults."""
        # if self.backtest_mode: # This condition is always true now
        return {
            "balance": self.backtest_initial_balance,
            "equity": self.backtest_initial_balance, 
            "margin": 0.0, # Simplified for backtesting; actual margin not tracked here
            "free_margin": self.backtest_initial_balance 
        }
        # else:
            # Live mode logic would go here
            # try:
            #     if self.mt5_handler and self.mt5_handler.initialized:
            #         return self.mt5_handler.get_account_info()
                
            #     logger.warning("MT5Handler not available or not initialized for account_info in live mode.")
            #     return {} # Or raise error
            # except Exception as e:
            #     logger.error(f"Error getting account info in live mode: {str(e)}")
            #     return {}
            
    def set_mt5_handler(self, mt5_handler: 'MT5Handler') -> None:
        """
        Set the MT5Handler instance. No-op in backtest_risk_manager.py.
        """
        if not self.backtest_mode: # Only relevant if live mode is ever re-enabled here
            logger.info("Setting MT5Handler in RiskManager (live mode context)")
            self.mt5_handler = mt5_handler
        else:
            logger.debug("set_mt5_handler called in backtest mode. No operation performed.")

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

        # --- Symbol Info Acquisition and Basic Validation ---
        # These will be populated based on mode (backtest or live)
        min_lot_from_symbol: float = 0.01
        contract_size_from_symbol: float = 1.0 
        volume_step_from_symbol: float = 0.01
        point_from_symbol: float = 0.00001 
        tick_value_acc_currency_from_symbol: Optional[float] = None
        # Use symbol-specific max volume if available, else global config max_lot_size
        # This will be assigned from symbol_props.get('volume_max', self.max_lot_size) or getattr(symbol_props, 'volume_max', self.max_lot_size)
        volume_max_from_symbol: float = self.max_lot_size 
        # symbol_props will hold the MT5 SymbolInfo object or the backtest dictionary
        raw_symbol_props: Optional[Any] = None 

        if self.backtest_mode:
            backtest_details = self.backtest_symbol_info.get(symbol)
            if not backtest_details:
                logger.warning(f"Backtest mode: Symbol details for {symbol} not found. Trade invalid.")
                return {'is_valid': False, 'reason': f"Backtest symbol info for {symbol} not found", 'position_size': 0.0}
            
            raw_symbol_props = backtest_details
            min_lot_from_symbol = backtest_details.get('volume_min', 0.01)
            contract_size_from_symbol = backtest_details.get('trade_contract_size', 1.0)
            volume_step_from_symbol = backtest_details.get('volume_step', 0.01)
            point_from_symbol = backtest_details.get('point', 0.00001)
            volume_max_from_symbol = backtest_details.get('volume_max', self.max_lot_size)
            tick_value_acc_currency_from_symbol = backtest_details.get('tick_value_in_account_currency_per_lot')
            
            if point_from_symbol is None or point_from_symbol == 0: # More robust check
                 logger.error(f"Backtest: 'point' for {symbol} is missing or zero. Point: {point_from_symbol}")
                 return {'is_valid': False, 'reason': f"Backtest: 'point' for {symbol} missing or zero", 'position_size': 0.0}
            if contract_size_from_symbol is None or contract_size_from_symbol == 0: # More robust check
                 logger.error(f"Backtest: 'trade_contract_size' for {symbol} is missing or zero. CS: {contract_size_from_symbol}")
                 return {'is_valid': False, 'reason': f"Backtest: 'trade_contract_size' for {symbol} missing or zero", 'position_size': 0.0}

            if tick_value_acc_currency_from_symbol is None: 
                # Calculate if essential components (point, contract_size) are valid
                tick_value_acc_currency_from_symbol = contract_size_from_symbol * point_from_symbol
                logger.info(f"Backtest: Calculated 'tick_value_in_account_currency_per_lot' for {symbol} as {tick_value_acc_currency_from_symbol} (CS: {contract_size_from_symbol}, Point: {point_from_symbol})")
            
            if tick_value_acc_currency_from_symbol == 0: # Check after potential calculation
                logger.error(f"Backtest: 'tick_value_in_account_currency_per_lot' for {symbol} is zero. Cannot calculate risk. TV: {tick_value_acc_currency_from_symbol}")
                return {'is_valid': False, 'reason': f"Backtest: tick_value for {symbol} is zero", 'position_size': 0.0}

            # Example: if not backtest_details.get('allow_trading', True):
            #     return {'is_valid': False, 'reason': f"Trading for {symbol} disabled in backtest config", 'position_size': 0.0}

        elif self.mt5_handler and self.mt5_handler.initialized:
            mt5_symbol_info_obj = self.mt5_handler.get_symbol_info(symbol)
            if not mt5_symbol_info_obj:
                logger.warning(f"Live mode: Could not retrieve symbol info for {symbol}. Trade invalid.")
                return {'is_valid': False, 'reason': f"Live symbol info for {symbol} not found", 'position_size': 0.0}
            
            if hasattr(mt5_symbol_info_obj, 'trade_mode') and mt5_symbol_info_obj.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                 logger.warning(f"Live mode: Trading for {symbol} is disabled by broker (trade_mode={mt5_symbol_info_obj.trade_mode}).")
                 return {'is_valid': False, 'reason': f"Trading for {symbol} is disabled by broker.", 'position_size': 0.0}
            
            raw_symbol_props = mt5_symbol_info_obj
            min_lot_from_symbol = getattr(mt5_symbol_info_obj, 'volume_min', 0.01)
            contract_size_from_symbol = getattr(mt5_symbol_info_obj, 'trade_contract_size', 1.0)
            volume_step_from_symbol = getattr(mt5_symbol_info_obj, 'volume_step', 0.01)
            point_from_symbol = getattr(mt5_symbol_info_obj, 'point', 0.00001)
            volume_max_from_symbol = getattr(mt5_symbol_info_obj, 'volume_max', self.max_lot_size)
            tick_value_acc_currency_from_symbol = getattr(mt5_symbol_info_obj, 'tick_value', None)

            if tick_value_acc_currency_from_symbol is None or tick_value_acc_currency_from_symbol == 0:
                logger.error(f"Live mode: tick_value for {symbol} is missing or zero ({tick_value_acc_currency_from_symbol}). Cannot proceed.")
                return {'is_valid': False, 'reason': f"Live mode: tick_value missing or zero for {symbol}", 'position_size': 0.0}
        else:
            logger.error("MT5 handler not available or not initialized. Cannot validate trade without symbol information.")
            return {'is_valid': False, 'reason': "MT5 handler unavailable for critical symbol info", 'position_size': 0.0}
        # --- End Symbol Info Acquisition ---

        # Basic trade parameter validation (already present, good)
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

        has_take_profit = 'take_profit_price' in trade or 'take_profit' in trade
        if not has_take_profit:
            logger.error("Trade validation failed: Take profit (take_profit_price or take_profit) is missing.")
            return {'is_valid': False, 'reason': "Take profit key (take_profit_price or take_profit) is missing", 'position_size': 0.0}

        entry_val = trade.get('entry_price') 
        stop_val = trade.get('stop_loss_price', trade.get('stop_loss'))
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
        # min_lot_from_symbol, contract_size_from_symbol, volume_step_from_symbol, volume_max_from_symbol are now set.

        if self.use_fixed_lot_size:
            position_size = self.fixed_lot_size
            position_size = max(min_lot_from_symbol, position_size)
            position_size = min(position_size, volume_max_from_symbol) # Use symbol specific max first
            if volume_step_from_symbol > 0:
                position_size = round(position_size / volume_step_from_symbol) * volume_step_from_symbol
            position_size = round(position_size, 8) # Standard rounding
            position_size = min(position_size, self.max_lot_size) # Then apply global max lot cap
            logger.info(f"Using fixed lot size for {symbol}: {position_size} (after normalization)")
        elif requested_size > 0:
            position_size = requested_size
            position_size = max(min_lot_from_symbol, position_size)
            position_size = min(position_size, volume_max_from_symbol) # Use symbol specific max first
            if volume_step_from_symbol > 0:
                position_size = round(position_size / volume_step_from_symbol) * volume_step_from_symbol
            position_size = round(position_size, 8) # Standard rounding
            position_size = min(position_size, self.max_lot_size) # Then apply global max lot cap
            logger.info(f"Using requested position_size (adjusted): {position_size}")
        else:
            try:
                risk_percentage_for_calc = self.max_risk_per_trade * 100.0
                position_size = self.calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=risk_percentage_for_calc, 
                    entry_price=entry,
                    stop_loss_price=stop,
                    symbol=symbol # calculate_position_size is now backtest-aware
                )
                # calculate_position_size already handles normalization and clamping based on mode.
                logger.info(f"Calculated position size based on risk: {position_size}")
            except (InvalidRiskParameterError, RiskCalculationError) as e:
                logger.error(f"Trade validation failed due to error in position sizing for {symbol}: {str(e)}")
                return {'is_valid': False, 'reason': f"Position sizing error: {str(e)}", 'position_size': 0.0}
            except Exception as e:
                logger.error(f"Unexpected error calculating position size for {symbol}: {str(e)}. Trade invalid.")
                return {'is_valid': False, 'reason': f"Unexpected position sizing error: {str(e)}", 'position_size': 0.0}
        
        if position_size <= 0:
            logger.warning(f"Initial position size for {symbol} is {position_size}. Trade invalid.")
            return {'is_valid': False, 'reason': f"Initial position size is not positive: {position_size}", 'position_size': 0.0}

        # --- Portfolio Level Risk Assessment ---
        logger.info(f"Performing portfolio level risk assessment for symbol: {symbol}")
        # calculate_portfolio_risk_limits will need its own backtest_mode adaptations.
        # It might need raw_symbol_props or specific values from it.
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
            if self.backtest_mode and isinstance(raw_symbol_props, dict): # In backtest, raw_symbol_props is a dict
                # Use 'current_price' from backtest_symbol_info (via raw_symbol_props) if available, else fallback to entry price
                current_price_from_portfolio_calc = raw_symbol_props.get('current_price', entry) 
                logger.info(f"Backtest: Using current_price {current_price_from_portfolio_calc} from raw_symbol_props (backtest_symbol_info) or entry price for portfolio check.")
            # elif not self.backtest_mode and raw_symbol_props is not None: # Live mode logic removed
            #      ...

            if current_price_from_portfolio_calc == 0.0: # Check again if still zero
                 logger.warning(f"Trade validation failed for {symbol}: Current price for portfolio calculation is zero even after fallbacks.")
                 return {'is_valid': False, 'reason': "Current price for portfolio calculation is zero", 'position_size': 0.0}
            
        logger.info(f"Portfolio analysis for {symbol}: Remaining Cash Limit: {remaining_cash_limit_from_portfolio:.2f}, Current Price: {current_price_from_portfolio_calc:.5f}")
        
        # --- Apply Portfolio Limit to Position Size ---
        # contract_size_from_symbol is already set correctly from the top block
        if self.use_fixed_lot_size:
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
                # Re-normalize after portfolio limit adjustment
                position_size = max(min_lot_from_symbol, position_size)
                position_size = min(position_size, volume_max_from_symbol) # Use symbol specific max
                if volume_step_from_symbol > 0:
                    position_size = round(position_size / volume_step_from_symbol) * volume_step_from_symbol
                position_size = round(position_size, 8)
                position_size = min(position_size, self.max_lot_size) # Global max cap
                
                if position_size < min_lot_from_symbol:
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
       
        # Final risk amount check using the determined position_size
        if stop != 0 and entry != 0: 
            risk_value_of_trade = 0.0
            # tick_value_acc_currency_from_symbol and point_from_symbol are now reliably set at the top.
            if tick_value_acc_currency_from_symbol is not None and tick_value_acc_currency_from_symbol > 0 and point_from_symbol > 1e-9:
                stop_loss_points_final = abs(entry - stop) / point_from_symbol
                risk_value_of_trade = position_size * stop_loss_points_final * tick_value_acc_currency_from_symbol
                
                max_acceptable_risk_amount = account_balance * self.max_risk_per_trade 
                if risk_value_of_trade > max_acceptable_risk_amount:
                    logger.warning(f"Trade validation failed for {symbol}: Calculated risk ({risk_value_of_trade:.2f}) for position size {position_size} exceeds max risk per trade ({max_acceptable_risk_amount:.2f}).")
                    return {
                        'is_valid': False,
                        'reason': f"Calculated risk {risk_value_of_trade:.2f} exceeds max risk per trade {max_acceptable_risk_amount:.2f}",
                        'position_size': position_size # Return the size that was too risky
                    }
                logger.info(f"Final risk for trade {symbol} with size {position_size}: {risk_value_of_trade:.2f} (Limit: {max_acceptable_risk_amount:.2f})")
            else:
                logger.warning(f"Could not calculate final risk value for {symbol} due to missing tick_value or point_from_symbol. Skipping this check.")

        # Check for concurrent trades limit
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
            # --- Fixed Lot Size Handling ---
            if self.use_fixed_lot_size:
                target_lot_size = self.fixed_lot_size
                logger.info(f"Using fixed lot size of {target_lot_size} for {symbol} (before normalization).")

                if self.backtest_mode:
                    symbol_details = self.backtest_symbol_info.get(symbol)
                    if not symbol_details:
                        logger.warning(f"Backtest mode: Symbol details for {symbol} not found. Using raw fixed lot: {target_lot_size} capped by global self.max_lot_size.")
                        # Apply global max_lot_size from TRADING_CONFIG as a fallback safety
                        return min(target_lot_size, self.max_lot_size) 
                    
                    vol_min = symbol_details.get('volume_min', 0.01)
                    # Use symbol-specific max if available, else global config's max_lot_size
                    vol_max_symbol_specific = symbol_details.get('volume_max')
                    vol_max = vol_max_symbol_specific if vol_max_symbol_specific is not None else self.max_lot_size

                    vol_step = symbol_details.get('volume_step', 0.01)

                    position_size = max(target_lot_size, vol_min)
                    position_size = min(position_size, vol_max) # Apply effective max (symbol specific or global)
                    if vol_step > 0: # Ensure vol_step is positive to avoid division by zero
                        position_size = round(position_size / vol_step) * vol_step
                    else: # if vol_step is 0 or None, no rounding by step is possible
                        logger.warning(f"Volume step for {symbol} is {vol_step}. Cannot normalize by step. Size: {position_size}")

                    position_size = round(position_size, 8) # Final rounding
                    
                    # Final check against the absolute global max_lot_size from TRADING_CONFIG
                    position_size = min(position_size, self.max_lot_size)

                    logger.info(f"Backtest mode: Fixed lot size for {symbol} normalized to {position_size} (Target:{target_lot_size}, Min:{vol_min}, MaxEff:{vol_max}, Step:{vol_step}, GlobalMax: {self.max_lot_size})")
                    return position_size
                elif self.mt5_handler and self.mt5_handler.initialized:
                    # Normalize using MT5 handler, which also considers symbol's min/max lot and step
                    normalized_size = self.mt5_handler.normalize_volume(symbol, target_lot_size)
                    # Ensure it doesn't exceed the configured max_lot_size as normalize_volume might respect symbol.volume_max
                    position_size = min(normalized_size, self.max_lot_size) 
                    logger.info(f"Live mode: Fixed lot size for {symbol} normalized to {position_size}")
                    return position_size
                else:
                    logger.warning(f"Live mode: MT5 handler not available. Using raw fixed lot: {target_lot_size} capped by config max_lot_size.")
                    return min(target_lot_size, self.max_lot_size)

            # --- Risk-Based Lot Size Calculation ---
            self._validate_position_inputs(account_balance, risk_per_trade / 100.0, entry_price, stop_loss_price)
            
            risk_amount_account_currency = account_balance * (risk_per_trade / 100.0)
            stop_loss_price_distance = abs(entry_price - stop_loss_price)

            if stop_loss_price_distance <= 1e-9: # Effectively zero or negative
                msg = f"Invalid stop loss distance: {stop_loss_price_distance} for {symbol}. Entry: {entry_price}, SL: {stop_loss_price}"
                logger.error(msg)
                raise RiskCalculationError(msg)

            point_val = 0.0
            value_per_point_per_lot_acc_currency = 0.0
            vol_min = 0.01
            vol_max = self.max_lot_size # Default to global max lot size
            vol_step = 0.01

            if self.backtest_mode:
                symbol_details = self.backtest_symbol_info.get(symbol)
                if not symbol_details:
                    raise RiskCalculationError(f"Backtest mode: Symbol details for {symbol} not found in backtest_symbol_info.")
                
                point_val = symbol_details.get('point')
                contract_size = symbol_details.get('trade_contract_size')
                # tick_value can be pre-calculated in account currency for backtesting or derived
                tick_value_acc_curr = symbol_details.get('tick_value_in_account_currency_per_lot') 

                if point_val is None or point_val == 0: # Check for None or zero
                    raise RiskCalculationError(f"Backtest mode: Missing or zero 'point' for {symbol}. Point: {point_val}")
                if contract_size is None or contract_size == 0: # Check for None or zero
                    raise RiskCalculationError(f"Backtest mode: Missing or zero 'trade_contract_size' for {symbol}. CS: {contract_size}")

                if tick_value_acc_curr is None: 
                    # Fallback: Assume profit currency is account currency and calculate
                    logger.info(f"Backtest mode: 'tick_value_in_account_currency_per_lot' not found for {symbol}. Calculating from point and contract_size.")
                    value_per_point_per_lot_acc_currency = contract_size * point_val
                else:
                    value_per_point_per_lot_acc_currency = tick_value_acc_curr
                
                if value_per_point_per_lot_acc_currency == 0: # Check after assignment/calculation
                     raise RiskCalculationError(f"Backtest mode: 'tick_value_in_account_currency_per_lot' is zero for {symbol} after obtaining/calculating. TV: {value_per_point_per_lot_acc_currency}")

                vol_min = symbol_details.get('volume_min', 0.01)
                # Use symbol-specific max if available, else global config's max_lot_size
                vol_max_symbol_specific = symbol_details.get('volume_max')
                vol_max = vol_max_symbol_specific if vol_max_symbol_specific is not None else self.max_lot_size
                vol_step = symbol_details.get('volume_step', 0.01)

            elif self.mt5_handler and self.mt5_handler.initialized:
                symbol_info_mt5 = self.mt5_handler.get_symbol_info(symbol)
                if not symbol_info_mt5:
                    raise RiskCalculationError(f"Live mode: Could not retrieve symbol info for {symbol}.")
                
                point_val = symbol_info_mt5.point
                # MT5 symbol_info.tick_value is typically value of one tick for 1 lot in deposit currency
                value_per_point_per_lot_acc_currency = symbol_info_mt5.tick_value 
                vol_min = symbol_info_mt5.volume_min
                vol_max = symbol_info_mt5.volume_max # Symbol specific max
                vol_step = symbol_info_mt5.volume_step
            else:
                # This will only be hit if self.backtest_mode is somehow False AND mt5_handler is None/uninitialized.
                # Given self.backtest_mode is hardcoded to True, this path is logically unreachable.
                raise RiskCalculationError("RiskManager misconfiguration: Not in backtest_mode or MT5 handler missing for risk-based sizing.")

            if point_val <= 1e-9:
                raise RiskCalculationError(f"Point value for {symbol} is zero or too small: {point_val}")
            
            stop_loss_points = stop_loss_price_distance / point_val
            if stop_loss_points <= 1e-9:
                raise RiskCalculationError(f"Stop loss in points is zero or too small for {symbol}: {stop_loss_points}")
            if value_per_point_per_lot_acc_currency <= 1e-9:
                raise RiskCalculationError(f"Value per point per lot for {symbol} is zero or too small: {value_per_point_per_lot_acc_currency}")

            position_size = risk_amount_account_currency / (stop_loss_points * value_per_point_per_lot_acc_currency)
            
            # Normalize and clamp
            position_size = max(position_size, vol_min) # Apply min volume
            position_size = min(position_size, vol_max) # Apply symbol specific max volume
            position_size = min(position_size, self.max_lot_size) # Apply global max lot size config
            
            if vol_step > 0:
                position_size = round(position_size / vol_step) * vol_step
            
            position_size = round(position_size, 8) # Final rounding for precision (lots usually 2dp, but steps can be smaller)

            if position_size < vol_min: # After all rounding, ensure it's not less than true min
                position_size = vol_min
                logger.info(f"Adjusted position size to symbol's minimum after rounding: {position_size} for {symbol}")

            logger.info(f"Calculated position size: {position_size} lots for {symbol} (Risk: {risk_per_trade}%, SL p_dist: {stop_loss_price_distance}, SL pts: {stop_loss_points:.2f})")
            return position_size
            
        except (InvalidRiskParameterError, InsufficientBalanceError) as e:
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
        cash: float = 0.0
        mt5_positions: List[Dict[str, Any]] = [] # Initialize as empty list

        if self.backtest_mode:
            logger.info("Portfolio risk limits in backtest mode.")
            cash = self.backtest_initial_balance # Use initial balance for backtest risk calcs
            mt5_positions = [] # In backtest mode, there are no live MT5 positions to fetch.
                               # Simulated open trades are in self.open_trades, but this function is typically
                               # called for pre-trade validation, so we assume an empty set of external positions.
            logger.debug(f"Backtest mode: Cash set to {cash}, live/external positions (mt5_positions) considered empty for this pre-trade check context.")
        else: # Corresponds to `if self.backtest_mode:`, should be unreachable if backtest_mode is always True.
            logger.error("RiskManager misconfiguration: Not in backtest_mode or MT5 handler missing for portfolio risk calculation.")
            for ticker in tickers_to_analyze:
                risk_analysis[ticker] = {"current_price": 0.0, "remaining_position_limit": 0.0, "reasoning": {"error": "RiskManager misconfigured for portfolio calc"}}
            return risk_analysis

        # Step 3: Determine all unique tickers involved (to analyze + existing positions from mt5_positions)
        all_relevant_tickers = set(tickers_to_analyze)
        # mt5_positions is empty in backtest_mode here, so this loop won't add anything.
        for pos in mt5_positions: # This loop will do nothing in backtest_mode as mt5_positions is [].
            if pos.get("symbol") and isinstance(pos.get("symbol"), str):
                all_relevant_tickers.add(pos["symbol"])
        logger.debug(f"All relevant tickers for price fetching (initially from tickers_to_analyze): {list(all_relevant_tickers)}")

        # Step 4: Fetch current prices for ALL relevant tickers
        for ticker in all_relevant_tickers:
            price = 0.0
            if self.backtest_mode:
                symbol_details = self.backtest_symbol_info.get(ticker)
                if symbol_details:
                    # Use 'current_price' if available in backtest_symbol_info, otherwise default to 0.0
                    # The backtester/data_loader should ideally populate 'current_price' if dynamic valuation is needed for strategies.
                    price = float(symbol_details.get('current_price', 0.0)) 
                    if price == 0.0:
                        logger.warning(f"Backtest mode: 'current_price' for {ticker} is 0 or missing in backtest_symbol_info. Risk limits may be inaccurate if price is needed here.")
                    current_prices[ticker] = price
                    logger.debug(f"Backtest mode: Price for {ticker} from backtest_symbol_info: {price}")
                else:
                    logger.warning(f"Backtest mode: No symbol details for {ticker} in backtest_symbol_info. Price set to 0 for portfolio calc.")
                    current_prices[ticker] = 0.0
            else: # Corresponds to `if self.backtest_mode:`, should be unreachable
                logger.error("Price fetching error: RiskManager misconfigured (not backtest_mode or MT5 handler missing).")
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
        