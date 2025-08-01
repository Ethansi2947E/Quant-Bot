import traceback
from typing import Dict, List, Any, Optional
from loguru import logger
import asyncio
import time
import hashlib

from src.risk_manager import RiskManager
from src.telegram.telegram_bot import TelegramBot
from src.mt5_handler import MT5Handler
from src.utils.position_manager import PositionManager

class SignalProcessor:
    """
    Handles signal processing and trade execution functionality.
    
    This class is responsible for:
    - Processing trading signals
    - Executing trades based on signals
    - Handling signals with existing positions
    - Validating signals against real-time MT5 data before execution
    """
    
    def __init__(self, mt5_handler=None, risk_manager=None, telegram_bot=None, config=None, position_manager=None):
        """
        Initialize the SignalProcessor.
        
        Args:
            mt5_handler: MT5Handler instance for executing trades
            risk_manager: RiskManager instance for position sizing
            telegram_bot: TelegramBot instance for notifications
            config: Configuration dictionary
            position_manager: PositionManager instance for multi-TP handling
        """
        self.mt5_handler = mt5_handler if mt5_handler is not None else MT5Handler()
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()
        self.telegram_bot = telegram_bot if telegram_bot else TelegramBot.get_instance()
        self.position_manager = position_manager if position_manager is not None else PositionManager()
        self.config = config or {}
        
        # State tracking
        self.active_trades = {}
        self.min_confidence = self.config.get("min_confidence", 0.6)  # Default to 60% confidence
        # Import TRADING_CONFIG for key settings to ensure we always use the current values
        from config.config import TRADING_CONFIG
        # Use TRADING_CONFIG directly for this sensitive setting
        self.allow_position_additions = TRADING_CONFIG.get("allow_position_additions", False)  # Default to NOT allowing position additions
        self.trading_enabled = self.config.get("trading_enabled", True)  # Default to enabled
        
        # Real-time validation settings
        self.validate_before_execution = self.config.get("validate_before_execution", True)
        # Increased from 0.0003 (0.03%) to 0.02 (2%) for more realistic market conditions
        self.price_validation_tolerance = self.config.get("price_validation_tolerance", 0.02)  # Default 2% tolerance
        self.tick_delay_tolerance = self.config.get("tick_delay_tolerance", 2.0)  # Maximum 2 seconds delay for ticks
        
        # Signal de-duplication
        self.processed_signals = {}  # Dictionary to track processed signals
        self.signal_expiry_time = self.config.get("signal_expiry_time", 300)  # Default 5 minutes
        
    def _generate_signal_hash(self, signal: Dict) -> str:
        """
        Generate a unique hash for a signal based on its key attributes.
        
        Args:
            signal: The signal dictionary
            
        Returns:
            str: A unique hash representing the signal
        """
        # Extract key values that define a unique signal
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        entry_price = signal.get("entry_price") or signal.get("entry", 0)
        stop_loss = signal.get("stop_loss", 0) 
        take_profit = signal.get("take_profit", 0)
        pattern = signal.get("pattern", "")
        timeframe = signal.get("timeframe", "")
        
        # Create a string representation of the signal
        signal_str = f"{symbol}:{direction}:{entry_price}:{stop_loss}:{take_profit}:{pattern}:{timeframe}"
        
        # Generate a hash from the string
        return hashlib.md5(signal_str.encode()).hexdigest()

    def _is_signal_recently_processed(self, signal: Dict) -> bool:
        """
        Check if a signal was recently processed to avoid duplicates.
        
        Args:
            signal: The signal dictionary
            
        Returns:
            bool: True if the signal was recently processed, False otherwise
        """
        signal_hash = self._generate_signal_hash(signal)
        
        # Check if we've already processed this signal
        if signal_hash in self.processed_signals:
            last_processed_time = self.processed_signals[signal_hash]
            current_time = time.time()
            
            # If signal was processed within the expiry window, consider it a duplicate
            if current_time - last_processed_time < self.signal_expiry_time:
                symbol = signal.get("symbol", "Unknown")
                direction = signal.get("direction", "Unknown")
                logger.warning(f"Duplicate signal detected for {symbol} {direction} - already processed {current_time - last_processed_time:.1f}s ago")
                return True
                
        # If we get here, this is a new signal or an expired one
        self.processed_signals[signal_hash] = time.time()
        return False
        
    # Clean up old signals to avoid memory leaks
    def _clean_processed_signals(self):
        """Remove expired signals from the processed_signals dictionary."""
        current_time = time.time()
        # Create a list of keys to remove
        expired_signals = [
            signal_hash for signal_hash, process_time in self.processed_signals.items()
            if current_time - process_time > self.signal_expiry_time
        ]
        
        # Remove expired signals
        for signal_hash in expired_signals:
            del self.processed_signals[signal_hash]
            
        if expired_signals:
            logger.debug(f"Cleaned up {len(expired_signals)} expired signals")
        
    async def initialize(self, config=None):
        """Initialize the SignalProcessor with the given configuration."""
        if config:
            self.config = config
            # Update derived values
            self.min_confidence = self.config.get("min_confidence", 0.2)
            # Always get the latest value from TRADING_CONFIG
            from config.config import TRADING_CONFIG
            self.allow_position_additions = TRADING_CONFIG.get("allow_position_additions", False)
            self.trading_enabled = self.config.get("trading_enabled", True)
            self.signal_expiry_time = self.config.get("signal_expiry_time", 300)
        
        # Initialize TelegramBot if needed
        if self.telegram_bot and hasattr(self.telegram_bot, 'initialize') and not getattr(self.telegram_bot, 'is_running', False):
            try:
                # Directly await TelegramBot initialization
                await self.telegram_bot.initialize(self.config)
                logger.info("Successfully initialized TelegramBot in SignalProcessor")
            except Exception as e:
                logger.error(f"Error initializing TelegramBot in SignalProcessor: {str(e)}")
        
        return True
        
    def set_mt5_handler(self, mt5_handler):
        """Set the MT5Handler instance after initialization."""
        self.mt5_handler = mt5_handler
        
    def set_risk_manager(self, risk_manager):
        """Set the RiskManager instance after initialization."""
        self.risk_manager = risk_manager
        
    def set_telegram_bot(self, telegram_bot):
        """Set the TelegramBot instance after initialization."""
        self.telegram_bot = telegram_bot
        
    def set_active_trades(self, active_trades):
        """Set the active trades dictionary after initialization."""
        self.active_trades = active_trades
    
    def validate_trade(self, signal: Dict, account_info: Optional[Dict] = None) -> Dict:
        """
        Validate a trade signal against risk management rules.
        
        Args:
            signal: Signal dictionary with trade details
            account_info: Optional account information dictionary
            
        Returns:
            Dictionary with validation result including:
                - valid (bool): Whether the trade is valid
                - reason (str): Reason if invalid
                - position_size (float): Final position size (canonical key)
                - adjusted_position_size (float): (Deprecated, for backward compatibility)
        """
        result = {"valid": True, "reason": ""}
        
        # Don't continue if risk manager is not available
        if not self.risk_manager:
            logger.warning("Risk manager not available for validation")
            result["valid"] = False
            result["reason"] = "Risk manager not available"
            return result
        
        symbol = signal.get('symbol', 'Unknown')
            
        # Get account info if not provided
        if not account_info and self.mt5_handler:
            # Try up to 3 times to get account info
            retries = 3
            for attempt in range(retries):
                account_info = self.mt5_handler.get_account_info()
                if account_info:
                    break
                    
                logger.warning(f"Attempt {attempt+1}/{retries}: Failed to get account info for validating {symbol} trade")
                
                # Check if MT5 is connected and attempt reconnection if needed
                if not self.mt5_handler.is_connected():
                    logger.warning("MT5 connection lost, attempting to reconnect...")
                    reconnect_success = self.mt5_handler.initialize()
                    if reconnect_success:
                        logger.info("Successfully reconnected to MT5")
                    else:
                        logger.error("Failed to reconnect to MT5")
                
                # Only sleep between retry attempts, not after the last attempt
                if attempt < retries - 1:
                    time.sleep(1)
        
        if not account_info:
            logger.error(f"Could not retrieve account info for validating {symbol} trade after multiple attempts")
            result["valid"] = False
            result["reason"] = "Could not retrieve account info for validation"
            return result
            
        account_balance = account_info.get("balance", 0)
        if account_balance <= 0:
            logger.warning(f"Invalid account balance for validation: {account_balance}")
            result["valid"] = False
            result["reason"] = f"Invalid account balance: {account_balance}"
            return result
            
        # Get open trades for context with retry logic
        open_trades = []
        if self.mt5_handler:
            # Try up to 2 times
            for attempt in range(2):
                open_trades = self.mt5_handler.get_open_positions()
                if open_trades is not None:  # Check if we got a valid response (empty list is fine)
                    break
                    
                logger.warning(f"Attempt {attempt+1}/2: Failed to get open positions for {symbol} validation")
                
                # Only retry once
                if attempt == 0:
                    time.sleep(1)
        
        # Try to validate through risk manager
        try:
            validation = self.risk_manager.validate_trade(
                trade=signal,
                account_balance=account_balance,
                open_trades=open_trades
            )
            
            # Process validation result
            if validation:
                # Check for 'is_valid' from the risk manager's response
                if "is_valid" in validation and isinstance(validation["is_valid"], bool):
                    result["valid"] = validation["is_valid"] # Store it as 'valid' in the local result dict for compatibility
                else:
                    result["valid"] = False
                
                # Copy other fields from validation, ensure 'position_size' is included
                # and 'reason' is also copied. We already handled 'is_valid' (by copying to 'valid').
                for key, value in validation.items():
                    if key != "is_valid": # Don't re-copy 'is_valid'
                        result[key] = value # This will copy 'position_size' and 'reason'
                
                # Ensure the 'adjusted_position_size' key is populated if 'position_size' exists, for downstream compatibility
                if "position_size" in result:
                    result["adjusted_position_size"] = result["position_size"]  # Deprecated, use 'position_size' instead
                
                # Log if signal doesn't comply with risk rules
                if not result.get("valid", False): # Check our local 'valid' flag
                    reason_message = result.get('reason', 'Unknown') # Assign to variable
                    logger.warning(f"[{symbol}] Signal doesn't comply with risk rules: {reason_message}") # Use variable in f-string
            
        except Exception as e:
            logger.warning(f"[{symbol}] Error validating trade: {str(e)}")
            logger.warning(traceback.format_exc())
            result["valid"] = False
            result["reason"] = f"Validation error: {str(e)}"
            
        return result
    
    async def process_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Process a list of signals and execute trades based on them.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            List of processed signals with results
        """
        process_start_time = time.time()
        logger.debug(f"[TIMING] 🔍 Starting signal processing at {process_start_time:.6f}")
        logger.info(f"Processing {len(signals)} signals")
        
        # Enhanced logging for position addition state
        from config.config import TRADING_CONFIG
        self.allow_position_additions = TRADING_CONFIG.get("allow_position_additions", False)
        logger.warning(f"🔄 SIGNAL PROCESSING: TRADING_CONFIG['allow_position_additions'] = {TRADING_CONFIG.get('allow_position_additions', False)}")
        logger.warning(f"🔄 SIGNAL PROCESSING: self.allow_position_additions = {self.allow_position_additions}")
        
        # Log current state of key settings
        logger.debug(f"Current settings: trading_enabled={self.trading_enabled}, allow_position_additions={self.allow_position_additions}")
        
        # CRITICAL NEW SAFETY CHECK: Get all existing positions at the start
        # to ensure we can correctly identify signals that would create additional positions
        all_positions_by_symbol = {}
        try:
            all_mt5_positions = self.mt5_handler.get_open_positions() if self.mt5_handler else []
            # Group by symbol
            for pos in all_mt5_positions:
                symbol = pos.get("symbol", "")
                if symbol:
                    if symbol not in all_positions_by_symbol:
                        all_positions_by_symbol[symbol] = []
                    all_positions_by_symbol[symbol].append(pos)
            
            logger.warning(f"🛑 SAFETY CHECK: Found {len(all_mt5_positions)} total positions across {len(all_positions_by_symbol)} symbols")
            for symbol, positions in all_positions_by_symbol.items():
                logger.warning(f"🛑 SAFETY CHECK: {symbol} has {len(positions)} open positions")
        except Exception as e:
            logger.error(f"Error during initial position check: {str(e)}")
            # Continue processing but with empty positions dictionary
            all_positions_by_symbol = {}
        
        if not signals:
            logger.debug("No signals to process")
            return []
            
        # Check if trading is enabled
        if not self.trading_enabled:
            logger.info("Trading is disabled. Skipping signal processing")
            for signal in signals:
                signal["status"] = "skipped"
                signal["error"] = "Trading is disabled"
                
            return signals

        symbols_to_process = set(signal.get("symbol") for signal in signals if signal.get("symbol"))
        logger.debug(f"[TIMING] 🔍 Symbols to process: {symbols_to_process}")
            
        # Check if MT5Handler is available
        if not self.mt5_handler:
            logger.error("MT5Handler not available. Cannot process signals")
            for signal in signals:
                signal["status"] = "error"
                signal["error"] = "MT5Handler not available"
                
            return signals
        
        mt5_check_start = time.time()
        # Check if connection to MT5 is lost
        if not self.mt5_handler.is_connected():
            logger.warning("MT5 connection lost before processing signals. Attempting to reconnect...")
            reconnect_start = time.time()
            reconnect_success = self.mt5_handler.initialize()
            reconnect_time = time.time() - reconnect_start
            logger.debug(f"[TIMING] 🔍 MT5 reconnection attempt took {reconnect_time:.4f}s")
            
            if not reconnect_success:
                logger.error("Failed to reconnect to MT5. Cannot process signals")
                for signal in signals:
                    signal["status"] = "error"
                    signal["error"] = "MT5 connection failed"
                return signals
            
            logger.info(f"Successfully reconnected to MT5 in {reconnect_time:.4f}s")
        mt5_check_time = time.time() - mt5_check_start
        logger.debug(f"[TIMING] 🔍 MT5 connection check took {mt5_check_time:.4f}s")
            
        logger.info(f"Processing {len(signals)} signals...")
        
        # Clean up expired signals
        cleanup_start = time.time()
        self._clean_processed_signals()
        cleanup_time = time.time() - cleanup_start
        logger.debug(f"[TIMING] 🔍 Signal cleanup took {cleanup_time:.4f}s")
        
        results = []
        
        # Process each signal
        for i, signal in enumerate(signals):
            signal_start_time = time.time()
            symbol = signal.get("symbol", "Unknown")
            direction = signal.get("direction", "Unknown")
            if direction is None:
                logger.warning(f"Missing direction for signal for {symbol}")
                continue
            logger.debug(f"[TIMING] 🔍 Processing signal {i+1}/{len(signals)}: {symbol} {direction} at {signal_start_time:.6f}")
            
            try:
                # Skip signals without confidence
                confidence_check_start = time.time()
                confidence = signal.get("confidence", 0)
                confidence_threshold = 0.50  # Minimum 66% confidence
                
                if confidence < confidence_threshold:
                    logger.debug(f"Signal for {symbol} has low confidence ({confidence:.2f}). Skipping.")
                    signal["status"] = "skipped"
                    signal["error"] = f"Low confidence: {confidence:.2f} < {confidence_threshold}"
                    results.append(signal)
                    confidence_check_time = time.time() - confidence_check_start
                    logger.debug(f"[TIMING] 🔍 Confidence check took {confidence_check_time:.4f}s - SKIPPED")
                    continue
                confidence_check_time = time.time() - confidence_check_start
                logger.debug(f"[TIMING] 🔍 Confidence check took {confidence_check_time:.4f}s - PASSED")
                
                # Skip signals for symbols that are not available in MT5
                symbol_check_start = time.time()
                if not self.mt5_handler.is_symbol_available(symbol):
                    logger.warning(f"Symbol {symbol} is not available in MT5. Skipping.")
                    signal["status"] = "skipped"
                    signal["error"] = f"Symbol {symbol} not available"
                    results.append(signal)
                    symbol_check_time = time.time() - symbol_check_start
                    logger.debug(f"[TIMING] 🔍 Symbol check took {symbol_check_time:.4f}s - SKIPPED")
                    continue
                symbol_check_time = time.time() - symbol_check_start
                logger.debug(f"[TIMING] 🔍 Symbol check took {symbol_check_time:.4f}s - PASSED")
                
                # Skip duplicate signals (same direction, symbol, and source within the last 5 minutes)
                duplicate_check_start = time.time()
                is_duplicate = self._is_signal_recently_processed(signal)
                if is_duplicate:
                    logger.debug(f"Duplicate signal for {symbol} {direction}. Skipping.")
                    signal["status"] = "skipped"
                    signal["error"] = f"Duplicate signal"
                    results.append(signal)
                    duplicate_check_time = time.time() - duplicate_check_start
                    logger.debug(f"[TIMING] 🔍 Duplicate check took {duplicate_check_time:.4f}s - SKIPPED (duplicate)")
                    continue
                duplicate_check_time = time.time() - duplicate_check_start
                logger.debug(f"[TIMING] 🔍 Duplicate check took {duplicate_check_time:.4f}s - PASSED")
                
                # Validate signal price against current market price
                price_validation_start = time.time()
                skip_price_validation = signal.get("skip_price_validation", False)
                
                if not skip_price_validation:
                    validation_result = await self._validate_signal_with_real_time_data(signal)
                    if not validation_result:
                        error_msg = "Price validation failed"
                        logger.warning(f"Signal validation failed for {symbol} {direction}: {error_msg}")
                        signal["status"] = "invalid"
                        signal["error"] = error_msg
                        results.append(signal)
                        price_validation_time = time.time() - price_validation_start
                        logger.debug(f"[TIMING] 🔍 Price validation took {price_validation_time:.4f}s - FAILED: {error_msg}")
                        continue
                    
                    logger.debug(f"Signal validation passed for {symbol} {direction}")
                    price_validation_time = time.time() - price_validation_start
                    logger.debug(f"[TIMING] 🔍 Price validation took {price_validation_time:.4f}s - PASSED")
                else:
                    logger.debug(f"Price validation skipped for {symbol} {direction} (skip_price_validation=True)")
                    price_validation_time = time.time() - price_validation_start
                    logger.debug(f"[TIMING] 🔍 Price validation check took {price_validation_time:.4f}s - SKIPPED (by request)")
                
                # Additional pre-execution validation could be added here
                
                # Right before trade execution, log position check
                logger.warning(f"🔄 PRE-EXECUTION CHECK: Signal #{i+1} for {symbol} {direction}")
                
                # ADDITIONAL SAFETY CHECK: Check against the positions we found at the start
                if not self.allow_position_additions and symbol in all_positions_by_symbol:
                    existing_symbol_positions = all_positions_by_symbol[symbol]
                    # Check if any positions are in the same direction
                    same_dir_positions = [p for p in existing_symbol_positions if 
                                         (p.get("type") == 0 and (direction or "").lower() == "buy") or
                                         (p.get("type") == 1 and (direction or "").lower() == "sell")]
                    
                    if same_dir_positions:
                        logger.warning(f"🚨 SAFETY BLOCK: Found {len(same_dir_positions)} existing positions for {symbol} in {direction} direction - additions disabled")
                        tickets = [p.get("ticket", "unknown") for p in same_dir_positions]
                        logger.warning(f"🚨 SAFETY BLOCK: Existing position tickets: {tickets}")
                        
                        # Mark signal as skipped
                        signal["status"] = "skipped"
                        signal["error"] = f"Position already exists in {direction} direction and additions are disabled"
                        results.append(signal)
                        continue
                
                # Get existing positions before executing trade
                existing_positions = []
                try:
                    all_positions = self.mt5_handler.get_open_positions()
                    existing_positions = [p for p in all_positions if p.get("symbol") == symbol]
                    
                    logger.warning(f"🔄 PRE-EXECUTION CHECK: Found {len(existing_positions)} existing positions for {symbol}")
                    
                    if existing_positions:
                        # Process with existing positions
                        result = await self.handle_signal_with_existing_positions(signal, existing_positions)
                        
                        # Update signal with result
                        if isinstance(result, dict):
                            if result.get("success") is False:
                                signal["status"] = "skipped"
                                signal["error"] = result.get("error", "Unknown error")
                                logger.warning(f"🔄 SIGNAL SKIPPED: {result.get('error', 'Unknown error')}")
                            else:
                                signal["status"] = result.get("status", "executed")
                                if "order" in result:
                                    signal["ticket"] = result.get("order")
                                    logger.warning(f"🔄 SIGNAL EXECUTED: Ticket {result.get('order')}")
                                
                        results.append(signal)
                        continue
                    else:
                        logger.warning(f"🔄 NO EXISTING POSITIONS: Creating new position for {symbol} {direction}")
                except Exception as e:
                    logger.error(f"Error checking existing positions before trade execution: {str(e)}")
                    # Continue with normal execution flow in case of error
                
                # Execute the trade
                trade_execution_start = time.time()
                logger.info(f"Executing trade for {symbol} {direction}")
                trade_result = await self.execute_trade_from_signal(signal)
                
                # Use the 'trade_result' dictionary for all post-execution logic
                if trade_result and trade_result.get("status") == "success":
                    signal["status"] = "executed"
                    signal["ticket"] = trade_result.get("order")
                    logger.info(f"Trade executed successfully for {symbol} {direction}. Ticket: {trade_result.get('order')}")
                    
                    # Store the processed signal to avoid duplicates
                    self._add_processed_signal(signal)
                    # --- RiskManager state update: trade opened ---
                    if self.risk_manager:
                        self.risk_manager.on_trade_opened(signal)
                    # --- NEW: Register trade with PositionManager for multi-TP handling ---
                    if self.position_manager and trade_result.get('ticket'):
                        self.position_manager.register_trade(
                            signal=signal,
                            trade_result=trade_result
                        )
                    else:
                        if not self.position_manager:
                            logger.debug("PositionManager not available, skipping trade registration.")
                        if not trade_result.get('ticket'):
                            logger.warning("Could not register trade for multi-TP: Ticket not found in trade result.")
                    # -------------------------------------------------------------------
                else:
                    signal["status"] = "failed"
                    signal["error"] = trade_result.get("message", "Unknown error")
                    error_code = trade_result.get("code", "Unknown code")
                    logger.error(f"Trade execution failed for {symbol} {direction}: {error_code} - {signal['error']}")
                
                trade_execution_time = time.time() - trade_execution_start
                execution_status = "SUCCESS" if trade_result and trade_result.get("status") == "success" else "FAILED"
                logger.debug(f"[TIMING] 🔍 Trade execution took {trade_execution_time:.4f}s - {execution_status}")
                
                # Add the processed signal to results
                results.append(signal)
                
            except Exception as e:
                logger.error(f"Error processing signal for {symbol}: {str(e)}")
                logger.debug(traceback.format_exc())
                signal["status"] = "error"
                signal["error"] = str(e)
                results.append(signal)
            
            signal_total_time = time.time() - signal_start_time
            signal_status = signal.get("status", "unknown")
            logger.debug(f"[TIMING] 🔍 Total processing time for signal {i+1} ({symbol} {direction}): {signal_total_time:.4f}s - Status: {signal_status}")
        
        process_total_time = time.time() - process_start_time
        success_count = sum(1 for s in results if s.get("status") == "executed")
        skip_count = sum(1 for s in results if s.get("status") == "skipped")
        fail_count = sum(1 for s in results if s.get("status") in ["invalid", "failed", "error"])
        
        logger.debug(f"[TIMING] 🔍 Total signal processing time: {process_total_time:.4f}s for {len(signals)} signals")
        logger.info(f"Signal processing summary: {success_count} executed, {skip_count} skipped, {fail_count} failed")
        
        return results
                
    async def execute_trade_from_signal(self, signal, symbol_info=None, is_addition=False):
        """
        Execute a trade based on a validated signal. This can be a market or limit order.
        """
        execution_start = time.time()
        symbol = signal.get('symbol', 'Unknown')
        direction = signal.get('direction', 'Unknown')
        order_type = signal.get('order_type', 'market') # Default to market for backward compatibility

        logger.debug(f"[TIMING] 💰 Starting {order_type.upper()} trade execution for {symbol} at {execution_start}")
        logger.warning(f"⚠️ {order_type.upper()} EXECUTION: Symbol={symbol}, Direction={direction}, is_addition={is_addition}")

        # --- PRE-VALIDATION ADJUSTMENT ---
        # Adjust SL and TP based on broker's minimum stop distance BEFORE final validation.
        try:
            current_tick = self.mt5_handler.get_last_tick(symbol)
            if not current_tick:
                raise ValueError("Could not retrieve current tick for pre-validation.")

            current_price = current_tick['ask'] if direction.lower() == 'buy' else current_tick['bid']
            min_stop_dist = self.mt5_handler.get_min_stop_distance(symbol)
            
            original_sl = float(signal.get('stop_loss', 0.0))
            
            # Check if the proposed stop loss is too close
            stop_dist_from_price = abs(current_price - original_sl)

            if stop_dist_from_price < min_stop_dist:
                logger.warning(f"[{symbol}] Original SL ({original_sl}) is too close to current price ({current_price}). Adjusting to meet min distance of {min_stop_dist}.")
                new_sl = current_price - min_stop_dist if direction.lower() == 'buy' else current_price + min_stop_dist
                
                # Now, recalculate TPs to maintain original R:R
                original_risk = abs(signal['entry_price'] - original_sl)
                new_risk = abs(current_price - new_sl)
                
                if original_risk > 0 and 'take_profits' in signal and signal['take_profits']:
                    risk_ratio = new_risk / original_risk
                    original_tps = signal['take_profits']
                    new_tps = []
                    for tp in original_tps:
                        original_reward = abs(tp - signal['entry_price'])
                        new_reward = original_reward * risk_ratio
                        new_tp = current_price + new_reward if direction.lower() == 'buy' else current_price - new_reward
                        new_tps.append(new_tp)
                    
                    logger.info(f"[{symbol}] Recalculated TPs from {signal['take_profits']} to {new_tps} to maintain R:R.")
                    signal['take_profits'] = new_tps
                
                signal['stop_loss'] = new_sl
                signal['entry_price'] = current_price # Update entry to current price for accuracy

        except Exception as e:
            logger.error(f"Failed during pre-validation adjustment for {symbol}: {e}")
            # Abort if we can't adjust properly
            return {'status': 'error', 'message': f'Pre-validation adjustment failed: {e}', 'code': 'ADJUSTMENT_FAILED'}


        # --- Perform trade validation (calls RiskManager.validate_trade) ---
        account_info_for_validation = self.mt5_handler.get_account_info() if self.mt5_handler else None
        validation_result = self.validate_trade(signal, account_info=account_info_for_validation)

        if not validation_result.get('valid'):
            logger.warning(f"Signal for {symbol} {direction} failed validation: {validation_result.get('reason')}")
            return {
                'status': 'error',
                'message': f"Signal validation failed: {validation_result.get('reason')}",
                'code': 'VALIDATION_FAILED'
            }
        
        position_size = validation_result.get('position_size')
        if position_size is None or position_size <= 0:
            logger.error(f"Invalid position size ({position_size}) after validation for {symbol} {direction}.")
            return {
                'status': 'error',
                'message': f"Invalid position size ({position_size}) after validation",
                'code': 'INVALID_SIZE_POST_VALIDATION'
            }
        logger.info(f"Using position size {position_size} for {symbol} {direction} from validation result.")

        # For market orders, we must ensure we aren't creating unwanted additional positions
        if order_type == 'market':
            all_positions = self.mt5_handler.get_open_positions() if self.mt5_handler else []
            symbol_positions = [p for p in all_positions if p.get("symbol") == symbol]
            same_direction_positions = [p for p in symbol_positions if
                                      (p.get("type") == 0 and (direction or "").lower() == "buy") or
                                      (p.get("type") == 1 and (direction or "").lower() == "sell")]

            if same_direction_positions and not is_addition:
                 if not self.allow_position_additions:
                    logger.warning(f"🚫 ADDITION FAILSAFE TRIGGERED: Blocking market order for {symbol} {direction} as it would create an additional position")
                    return {
                        'status': 'error',
                        'message': 'Blocked unmarked position addition - positions exist but additions disabled',
                        'code': 'UNMARKED_ADDITION_BLOCKED'
                    }

        try:
            if not self.mt5_handler.is_connected():
                logger.error("MT5 not connected for trade execution")
                return {'status': 'error', 'message': 'MT5 not connected'}

            # --- Extract signal parameters ---
            entry_price = signal.get('entry_price', 0)
            stop_loss = float(signal.get('stop_loss', 0.0))
            
            # --- Multi-TP Handling ---
            # Look for a list of TPs first, then fall back to a single TP
            take_profits_list = signal.get('take_profits') # This will be a list or None
            take_profit_single = float(signal.get('take_profit', 0.0)) # Fallback single value
            
            # For the order, we use the single TP if the list is absent, 
            # otherwise MT5Handler will use the first from the list.
            final_take_profit_for_order = take_profit_single if not take_profits_list else take_profits_list[0]

            logger.debug(f"Executing trade for {symbol} {direction}: size={position_size}, SL={stop_loss}, TP={final_take_profit_for_order}")
            
            # Execute the trade using MT5Handler
            trade_result = self.mt5_handler.place_market_order(
                symbol=symbol,
                order_type=direction,
                volume=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit_single, # Pass single TP for backward compatibility/clarity
                take_profits=take_profits_list, # Pass the list for new logic
                comment=f"{signal.get('strategy_name', 'N/A')}" # Pass strategy name in comment
            )

            if trade_result and 'ticket' in trade_result and trade_result['ticket'] > 0:
                ticket = trade_result['ticket']
                logger.info(f"✅ {order_type.upper()} order processed successfully: {symbol} {(direction or '')} {position_size} lots")
                if self.risk_manager:
                    self.risk_manager.on_trade_opened(signal)
                
                # Send a confirmation message via Telegram
                if self.telegram_bot:
                    order_type = "MARKET" if order_type == 'market' else "PENDING"
                    trade_details = self._build_trade_notification(signal, symbol, direction, entry_price, stop_loss, final_take_profit_for_order, position_size)
                    try:
                        # Correctly await the async send_message function
                        await self.telegram_bot.send_message(f"✅ {order_type.upper()} Order Placed\n\n{trade_details}")
                    except Exception as e:
                        logger.error(f"Failed to send Telegram confirmation for {symbol} {direction}: {e}")
                
                return {
                    'status': 'success',
                    'order': ticket,
                    'ticket': ticket,  # Added for compatibility with PositionManager
                    'message': f'{order_type.capitalize()} order placed successfully',
                    'time_taken': time.time() - execution_start
                }
            else:
                error_message = trade_result.get('message', f'Failed to place {order_type} order') if trade_result else 'Execution failed: MT5 handler returned no result'
                logger.error(f"❌ {order_type.upper()} execution failed: {error_message}")
                if self.telegram_bot:
                    await self.telegram_bot.send_message(
                        f"❌ {order_type.upper()} Execution Failed\n\n"
                        f"Symbol: {symbol}\n"
                        f"Direction: {(direction or '').upper()}\n"
                        f"Error: {error_message}"
                    )
                return {
                    'status': 'error',
                    'message': error_message,
                    'code': 'MT5_EXECUTION_ERROR',
                    'time_taken': time.time() - execution_start
                }
                
        except Exception as e:
            logger.error(f"Exception during {order_type} trade execution: {str(e)}")
            logger.error(traceback.format_exc())
            execution_time = time.time() - execution_start
            return {
                'status': 'error',
                'message': str(e),
                'code': 'Exception',
                'time_taken': execution_time
            }
            
    async def handle_signal_with_existing_positions(self, signal: Dict, existing_positions: List[Dict]) -> Dict:
        """
        Process a signal when there are already open positions for the symbol.
        
        Args:
            signal: Signal dictionary containing trade details
            existing_positions: List of existing position dictionaries
            
        Returns:
            Dict: Result of the operation
        """
        from config.config import TRADING_CONFIG
        
        # CRITICAL FIX: Always re-check the setting from TRADING_CONFIG
        self.allow_position_additions = TRADING_CONFIG.get("allow_position_additions", False)
        
        symbol = signal.get("symbol")
        direction = signal.get("direction", "").lower()
        
        # DEBUG: Enhanced position logging
        logger.warning(f"🔍 POSITION CHECK: Symbol={symbol}, Direction={direction}, allow_position_additions={self.allow_position_additions}")
        logger.warning(f"🔍 POSITION CHECK: Found {len(existing_positions)} existing positions for {symbol}")
        
        # Log each existing position
        for i, pos in enumerate(existing_positions):
            pos_symbol = pos.get("symbol", "unknown")
            pos_ticket = pos.get("ticket", "unknown")
            pos_type = "BUY" if pos.get("type", 0) == 0 else "SELL"
            pos_volume = pos.get("volume", 0.0)
            logger.warning(f"🔍 POSITION CHECK: Existing position #{i+1}: Symbol={pos_symbol}, Ticket={pos_ticket}, Type={pos_type}, Volume={pos_volume}")
        
        # Group positions by direction
        buy_positions = [p for p in existing_positions if p.get("type") == 0]  # MT5 type 0 = BUY
        sell_positions = [p for p in existing_positions if p.get("type") == 1]  # MT5 type 1 = SELL
        
        # Check if we have positions in the same direction as the signal
        same_direction_positions = buy_positions if direction == "buy" else sell_positions
        
        # Log position counts by direction
        logger.warning(f"🔍 POSITION CHECK: Found {len(buy_positions)} BUY positions and {len(sell_positions)} SELL positions")
        logger.warning(f"🔍 POSITION CHECK: Found {len(same_direction_positions)} positions in same direction as signal")
        
        if same_direction_positions:
            # We already have positions in this direction
            # Use the instance variable instead of reading from config each time
            # Always log the current value for debugging
            logger.warning(f"🔍 POSITION CHECK: Position additions setting: allow_position_additions={self.allow_position_additions}")
            
            if self.allow_position_additions:
                # Add to existing position if allowed
                logger.warning(f"✅ POSITION ADDITION ALLOWED: Adding to existing {direction} position for {symbol}")
                result = await self.execute_trade_from_signal(signal, is_addition=True)
                return result
            else:
                logger.warning(f"❌ POSITION ADDITION BLOCKED: Already have {direction} position for {symbol} and additions are disabled")
                return {"success": False, "error": "Position already exists in this direction and additions are disabled"}
        else:
            # We have positions but in the opposite direction
            opposite_positions = sell_positions if direction == "buy" else buy_positions
            
            if opposite_positions:
                logger.warning(f"❓ OPPOSITE DIRECTION POSITIONS: Signal conflicts with existing {len(opposite_positions)} opposite positions for {symbol}")
                # We could implement logic to close opposite positions here
                # For now, just log the conflict
                return {"success": False, "error": "Conflicting position exists in opposite direction"}
            
            # No positions in either direction for this symbol
            logger.warning(f"✅ NEW POSITION: No existing positions for {symbol} in either direction, creating new position")
            # Still execute the trade in the new direction if desired
            # This is commented out to prevent having positions in both directions
            # return await self.execute_trade_from_signal(signal)
            return {"success": False, "error": "Preventing position in both directions"}
            
    async def notify_trade_action(self, message: str) -> None:
        """
        Send notification about a trade action.
        
        Args:
            message: Notification message
        """
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_notification(message)
            except Exception as e:
                logger.warning(f"Failed to send notification: {str(e)}")
                
    def _get_position_type(self, position: Dict[str, Any]) -> str:
        """
        Get the position type (buy/sell) from a position dictionary.
        
        Args:
            position: Position data dictionary
            
        Returns:
            String indicating position type ("buy" or "sell")
        """
        position_type = position.get("type", 0)
        return "buy" if position_type == 0 else "sell"

    async def _validate_signal_with_real_time_data(self, signal: Dict) -> bool:
        """
        Validate the signal against real-time MT5 tick data before execution.
        
        Args:
            signal: The signal dictionary to validate
            
        Returns:
            bool: True if the signal is valid when compared to real-time data
        """
        validation_start = time.time()
        try:
            symbol = signal.get("symbol")
            direction = signal.get("direction")
            entry_price = signal.get("entry_price")
            
            logger.debug(f"[TIMING] 🔍 Starting price validation for {symbol} {direction} at price {entry_price}")
            
            if not symbol or not direction or not entry_price:
                logger.warning(f"Cannot validate signal missing essential data: {signal}")
                return False
                
            # Skip validation if signal has override flag
            if signal.get("skip_price_validation", False):
                logger.info(f"Skipping price validation for {symbol} {direction} as requested by signal")
                return True
                
            # Initialize variables for retry logic
            retry_count = 0
            max_retries = 3
            latest_tick = None
                
            # Try to get fresh tick data with retries
            while retry_count < max_retries:
                tick_fetch_start = time.time()
                # Get the latest tick from MT5
                latest_tick = self.mt5_handler.get_last_tick(symbol)
                tick_fetch_time = time.time() - tick_fetch_start
                
                logger.debug(f"[TIMING] 🔍 Tick fetch attempt {retry_count+1} for {symbol} took {tick_fetch_time:.4f}s")
                
                if not latest_tick:
                    logger.warning(f"Attempt {retry_count+1}/{max_retries}: Cannot get latest tick for {symbol}")
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        # Try to refresh symbol data and retry
                        refresh_start = time.time()
                        refresh_success = self.mt5_handler.is_symbol_available(symbol)
                        refresh_time = time.time() - refresh_start
                        
                        logger.debug(f"[TIMING] 🔍 Symbol refresh for {symbol} took {refresh_time:.4f}s: {'SUCCESS' if refresh_success else 'FAILED'}")
                        
                        if refresh_success:
                            logger.debug(f"Successfully refreshed symbol {symbol} in MarketWatch")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        logger.warning(f"Failed to get tick data for {symbol} after {max_retries} attempts, skipping validation")
                        validation_total = time.time() - validation_start
                        logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - SKIPPED (no tick data)")
                        return True  # Don't fail validation if we can't get tick data
                
                # Check tick freshness - ensure tick is recent
                now = time.time()
                tick_time = latest_tick.get('time', now)
                time_diff = now - tick_time
                
                logger.debug(f"[TIMING] 🔍 Tick age check for {symbol}: {time_diff:.4f}s old (tolerance: {self.tick_delay_tolerance:.4f}s)")
                
                if time_diff > self.tick_delay_tolerance:
                    logger.warning(f"Attempt {retry_count+1}/{max_retries}: Tick data for {symbol} is too old ({time_diff:.2f}s)")
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        # Try to refresh symbol and retry
                        refresh_start = time.time()
                        refresh_success = self.mt5_handler.is_symbol_available(symbol)
                        refresh_time = time.time() - refresh_start
                        
                        logger.debug(f"[TIMING] 🔍 Symbol refresh for {symbol} took {refresh_time:.4f}s: {'SUCCESS' if refresh_success else 'FAILED'}")
                        
                        if refresh_success:
                            logger.debug(f"Refreshed symbol {symbol} in MarketWatch")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        logger.warning(f"Tick data for {symbol} is still too old ({time_diff:.2f}s) after {max_retries} attempts")
                        if time_diff > 600:  # More than 10 minutes
                            logger.error(f"Tick data is severely outdated (>10 min). Symbol may be closed or unavailable.")
                            validation_total = time.time() - validation_start
                            logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - FAILED (stale tick)")
                            return False  # Fail validation for extremely stale data
                        validation_total = time.time() - validation_start
                        logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - PASSED (moderately stale tick)")
                        return True  # Continue with validation for moderately stale data
                
                # Break the loop if we have fresh tick data
                break
                
            # If we get here and still don't have tick data, skip validation
            if not latest_tick:
                logger.warning(f"No tick data available for {symbol}, skipping validation")
                validation_total = time.time() - validation_start
                logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - SKIPPED (no tick data)")
                return True
                
            # Get current bid/ask prices
            current_bid = latest_tick.get('bid', 0)
            current_ask = latest_tick.get('ask', 0)
            
            if current_bid == 0 or current_ask == 0:
                logger.warning(f"Invalid bid/ask prices for {symbol}: bid={current_bid}, ask={current_ask}")
                validation_total = time.time() - validation_start
                logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - PASSED (invalid prices)")
                return True  # Don't fail validation on missing prices, just warn
                
            # Calculate price difference based on signal direction
            price_to_compare = current_bid if direction.lower() == "sell" else current_ask
            price_diff_percent = abs(entry_price - price_to_compare) / price_to_compare
            
            # Log the validation check
            logger.debug(f"Validating {direction} signal for {symbol}. "
                        f"Signal price: {entry_price:.5f}, Current {direction} price: {price_to_compare:.5f}, "
                        f"Difference: {price_diff_percent:.5%}, Tolerance: {self.price_validation_tolerance:.5%}")
            
            # Validate the price is within tolerance
            if price_diff_percent <= self.price_validation_tolerance:
                logger.info(f"Signal for {symbol} {direction} validated: "
                          f"Signal price {entry_price:.5f} matches current price {price_to_compare:.5f} "
                          f"(diff: {price_diff_percent:.5%})")
                validation_total = time.time() - validation_start
                logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - PASSED")
                return True
            else:
                logger.warning(f"Signal for {symbol} {direction} rejected: "
                             f"Price discrepancy too large. Signal: {entry_price:.5f}, "
                             f"Current: {price_to_compare:.5f}, Diff: {price_diff_percent:.5%}, "
                             f"Max allowed: {self.price_validation_tolerance:.5%}")
                validation_total = time.time() - validation_start
                logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - FAILED (price discrepancy)")
                return False
                
        except Exception as e:
            validation_total = time.time() - validation_start
            logger.error(f"Error validating signal against real-time data: {str(e)}")
            logger.error(traceback.format_exc())
            logger.debug(f"[TIMING] 🔍 Validation for {symbol} completed in {validation_total:.4f}s - PASSED (error bypass)")
            return True  # Don't fail validation on error, just warn 

    def _add_processed_signal(self, signal: Dict) -> None:
        """
        Add a signal to the processed signals list to avoid duplicates.
        
        Args:
            signal: The signal dictionary to add to processed list
        """
        signal_hash = self._generate_signal_hash(signal)
        # Store timestamp for backward compatibility
        self.processed_signals[signal_hash] = time.time()
        logger.debug(f"Added signal hash {signal_hash} to processed signals")

    def _recalculate_tp_sl_for_price_deviation(self, signal: Dict, current_price: float) -> Dict:
        """
        Recalculate take profit and stop loss when there's a significant deviation between
        signal price and execution price, while maintaining the original risk-to-reward ratio.
        
        Args:
            signal: The original signal dictionary
            current_price: The current market price for execution
            
        Returns:
            Dict: Updated signal with recalculated TP and SL
        """
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        original_entry = signal.get("entry_price")
        original_sl = signal.get("stop_loss")
        original_tp = signal.get("take_profit")
        
        # Exit early if any of the essential values is missing
        if original_entry is None or original_sl is None or original_tp is None:
            logger.warning(f"Cannot recalculate TP/SL for {signal.get('symbol', 'Unknown')} - missing values")
            return signal
        # Ensure numeric type
        original_entry = float(original_entry)
        original_sl = float(original_sl)
        original_tp = float(original_tp)
        if (direction or "").lower() == "buy":
            original_risk = original_entry - original_sl
            original_reward = original_tp - original_entry
        else:  # sell
            original_risk = original_sl - original_entry
            original_reward = original_entry - original_tp
            
        # Calculate original risk-to-reward ratio
        if original_risk == 0:
            logger.warning(f"Cannot recalculate TP/SL for {symbol} - original risk is zero")
            return signal
            
        original_rr_ratio = original_reward / original_risk
        
        # Calculate new SL and TP based on current price, maintaining the same R:R ratio
        updated_signal = signal.copy()
        if (direction or "").lower() == "buy":
            new_sl = current_price - original_risk
            new_tp = current_price + (original_risk * original_rr_ratio)
        else:  # sell
            new_sl = current_price + original_risk
            new_tp = current_price - (original_risk * original_rr_ratio)
            
        # Log the adjustments
        price_deviation_pct = abs(current_price - original_entry) / original_entry * 100
        logger.info(f"Price deviation for {symbol} {direction}: {price_deviation_pct:.2f}% from signal price")
        logger.info(f"Recalculating TP/SL for {symbol} {direction}:")
        logger.info(f"  Original entry: {original_entry:.5f}, Current price: {current_price:.5f}")
        logger.info(f"  Original SL: {original_sl:.5f} → New SL: {new_sl:.5f}")
        logger.info(f"  Original TP: {original_tp:.5f} → New TP: {new_tp:.5f}")
        logger.info(f"  Maintaining original R:R ratio of {original_rr_ratio:.2f}")
        
        # Update the signal with new values
        updated_signal["entry_price"] = current_price
        updated_signal["stop_loss"] = new_sl
        updated_signal["take_profit"] = new_tp
        
        return updated_signal 

    def _build_trade_notification(self, signal: Dict, symbol: str, direction: str, entry_price: float, stop_loss: float, take_profit: float, position_size: float) -> str:
        """
        Build a comprehensive trade notification message based on the signal format.
        
        Args:
            signal: The signal dictionary
            symbol: The trading symbol
            direction: The trading direction
            entry_price: The entry price
            stop_loss: The stop loss price
            take_profit: The take profit price
            position_size: The trading position size
            
        Returns:
            str: The formatted trade notification message
        """
        # Extract relevant information from the signal
        strategy_name = signal.get('strategy_name') or signal.get('strategy') or signal.get('source') or 'Unknown'
        confidence = signal.get('confidence', 0.0)
        confidence_pct = confidence * 100 if isinstance(confidence, (float, int)) else 0.0
        reason_text = signal.get('reason', 'N/A')
        
        # Build the notification based on the signal format
        if 'score_details' in signal and isinstance(signal.get('score_details'), dict):
            # BreakoutReversalStrategy scoring format
            sd = signal['score_details']
            final_score_pct = signal.get('score', 0.0) * 100 if isinstance(signal.get('score'), (float, int)) else 0.0
            level_strength_pct = sd.get('level_strength', 0.0) * 100
            volume_quality_pct = sd.get('volume_quality', 0.0) * 100
            pattern_reliability_pct = sd.get('pattern_reliability', 0.0) * 100
            trend_alignment_pct = sd.get('trend_alignment', 0.0) * 100
            risk_reward_pct = sd.get('risk_reward', 0.0) * 100
            
            # Get the detailed reasoning if available
            detailed_reasoning = signal.get('detailed_reasoning', [])
            detailed_analysis = ""
            if detailed_reasoning and isinstance(detailed_reasoning, list):
                detailed_analysis = "\n".join([f"• {reason}" for reason in detailed_reasoning])
            
            # If no detailed reasoning is available, use the original reason
            if not detailed_analysis:
                detailed_analysis = signal.get('reason', 'N/A')
            
            # Use strategy_name from signal if available
            strategy_name = signal.get('strategy_name') or signal.get('strategy') or 'Unknown'
            # Check for multiple TPs and format them if they exist
            take_profits = signal.get('take_profits')
            if take_profits and isinstance(take_profits, list):
                tp_text = ", ".join([f"TP{i+1}: {tp:.5f}" for i, tp in enumerate(take_profits)])
            else:
                tp_text = f"{take_profit:.5f}" # Fallback to single TP
            
            trade_details = (
                f"🔸 Strategy: {strategy_name}\n"
                f"🔹 Symbol: {symbol}\n"
                f"🔹 Direction: {(direction or '').upper()}\n"
                f"🔹 Entry: {entry_price:.5f}\n"
                f"🔹 Stop Loss: {stop_loss:.5f}\n"
                f"🔹 Take Profit: {tp_text}\n"
                f"🔹 Size: {position_size} lots\n\n"
                f"📊 Confidence: {confidence_pct:.1f}%\n"
                f"📊 Signal Quality: {self._get_score_emoji(final_score_pct)} ({final_score_pct:.1f}%)\n"
                f"• Level Strength: {level_strength_pct:.1f}% (30% weight)\n"
                f"• Volume Quality: {volume_quality_pct:.1f}% (20% weight)\n"
                f"• Pattern Reliability: {pattern_reliability_pct:.1f}% (20% weight)\n"
                f"• Trend Alignment: {trend_alignment_pct:.1f}% (20% weight)\n"
                f"• Risk-Reward: {risk_reward_pct:.1f}% (10% weight)\n\n"
                f"📝 Analysis:\n{detailed_analysis}"
            )
            
            # Add any special bonuses that were applied
            bonuses = []
            if signal.get('_volume_profile_bonus', False):
                bonuses.append("📈 Volume Profile Node Bonus (+0.07)")
            if signal.get('_atr_bonus', 0) > 0:
                bonuses.append("📏 Optimal Stop Placement Bonus (+0.1)")
            elif signal.get('_atr_bonus', 0) < 0:
                bonuses.append("⚠️ Suboptimal Stop Placement Penalty (-0.1)")
            if signal.get('consolidation_bonus', False):
                bonuses.append("📦 Inside Consolidation Zone Bonus (+0.05)")
            
            if bonuses:
                trade_details += "\n\n🎯 Applied Score Adjustments:\n" + "\n".join(bonuses)
            
        elif signal.get('signal_quality') is not None:
            # Classic confluence-style notification
            signal_quality = signal.get('signal_quality', 0.0)
            pattern_score = signal.get('pattern_score', 0.0)
            confluence_score = signal.get('confluence_score', 0.0)
            volume_score = signal.get('volume_score', 0.0)
            recency_score = signal.get('recency_score', 0.0)
            # Format as percentages
            signal_quality_pct = signal_quality * 100
            pattern_score_pct = pattern_score * 100
            confluence_score_pct = confluence_score * 100
            volume_score_pct = volume_score * 100
            recency_score_pct = recency_score * 100
            # Use detailed_reasoning if present and non-empty, else fallback to reason
            detailed_reasoning = signal.get('detailed_reasoning', [])
            if detailed_reasoning and isinstance(detailed_reasoning, list):
                analysis_text = "\n".join(str(r) for r in detailed_reasoning)
            else:
                analysis_text = signal.get('reason', 'N/A')
            # Check for multiple TPs and format them if they exist
            take_profits = signal.get('take_profits')
            if take_profits and isinstance(take_profits, list):
                tp_text = ", ".join([f"TP{i+1}: {tp:.5f}" for i, tp in enumerate(take_profits)])
            else:
                tp_text = f"{take_profit:.5f}" # Fallback to single TP
            
            trade_details = (
                f"🔸 Strategy: {strategy_name}\n"
                f"🔹 Symbol: {symbol}\n"
                f"🔹 Direction: {(direction or '').upper()}\n"
                f"🔹 Entry: {entry_price:.5f}\n"
                f"🔹 Stop Loss: {stop_loss:.5f}\n"
                f"🔹 Take Profit: {tp_text}\n"
                f"🔹 Size: {position_size} lots\n\n"
                f"📊 Confidence: {signal.get('confidence', 0) * 100:.1f}%\n"
                f"📊 Signal Quality: {self._get_score_emoji(signal_quality_pct)} ({signal_quality_pct:.1f}%)\n"
                f"• Pattern: {pattern_score_pct:.1f}% (40% weight)\n"
                f"• Confluence: {confluence_score_pct:.1f}% (40% weight)\n"
                f"• Volume: {volume_score_pct:.1f}% (10% weight)\n"
                f"• Recency: {recency_score_pct:.1f}% (10% weight)\n\n"
                f"📝 Analysis:\n{analysis_text}"
            )
        elif signal.get('score_01') is not None:
            # PriceActionSRStrategy format with score_01 and score_breakdown
            score_01 = signal.get('score_01', 0.0)
            score_breakdown = signal.get('score_breakdown', {})
            score_01_pct = score_01 * 100
            
            # Extract breakdown components
            pattern_score = score_breakdown.get('pattern', 0.0) * 100
            wick_score = score_breakdown.get('wick', 0.0) * 100
            volume_score = score_breakdown.get('volume', 0.0) * 100
            risk_reward_score = score_breakdown.get('risk_reward', 0.0) * 100
            zone_score = score_breakdown.get('zone_strength', 0.0) * 100
            
            # Get pattern and zone information
            pattern = signal.get('pattern', 'Unknown')
            zone = signal.get('zone', 0.0)
            zone_touches = signal.get('zone_touches', 0)
            
            # Check for multiple TPs and format them if they exist
            take_profits = signal.get('take_profits')
            if take_profits and isinstance(take_profits, list):
                tp_text = ", ".join([f"TP{i+1}: {tp:.5f}" for i, tp in enumerate(take_profits)])
            else:
                tp_text = f"{take_profit:.5f}" # Fallback to single TP
            
            trade_details = (
                f"🔸 Strategy: {strategy_name}\n"
                f"🔹 Symbol: {symbol}\n"
                f"🔹 Direction: {(direction or '').upper()}\n"
                f"🔹 Entry: {entry_price:.5f}\n"
                f"🔹 Stop Loss: {stop_loss:.5f}\n"
                f"🔹 Take Profit: {tp_text}\n"
                f"🔹 Size: {position_size} lots\n\n"
                f"📊 Confidence: {confidence_pct:.1f}%\n"
                f"📊 Signal Quality: {self._get_score_emoji(score_01_pct)} ({score_01_pct:.1f}%)\n"
                f"• Pattern: {pattern_score:.1f}% (30% weight)\n"
                f"• Wick Rejection: {wick_score:.1f}% (25% weight)\n"
                f"• Volume: {volume_score:.1f}% (25% weight)\n"
                f"• Risk-Reward: {risk_reward_score:.1f}% (15% weight)\n"
                f"• Zone Strength: {zone_score:.1f}% (5% weight)\n\n"
                f"📊 Zone: {zone:.5f} (touched {zone_touches} times)\n"
                f"📊 Pattern: {pattern}\n\n"
                f"📝 Analysis:\n{reason_text}"
            )
        else:
            # Enhanced basic format - extract as much useful information as possible
            strategy_name = signal.get('strategy_name') or signal.get('strategy') or signal.get('source') or 'Unknown'
            confidence_pct = signal.get('confidence', 0.0) * 100 if isinstance(signal.get('confidence'), (float, int)) else 0.0
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss) if stop_loss else 0
            reward = abs(take_profit - entry_price) if take_profit else 0
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Extract pattern information
            pattern = signal.get('pattern', 'Price Action')
            
            # Build enhanced analysis text
            analysis_parts = []
            
            # Add the basic reason
            if reason_text and reason_text != 'N/A':
                analysis_parts.append(reason_text)
            
            # Add pattern details if available
            if pattern and pattern != 'Price Action':
                analysis_parts.append(f"Pattern: {pattern}")
            
            # Add timeframe information
            timeframe = signal.get('timeframe', 'Unknown')
            if timeframe != 'Unknown':
                analysis_parts.append(f"Timeframe: {timeframe}")
            
            # Add technical metrics if available
            technical_metrics = signal.get('technical_metrics', {})
            if technical_metrics:
                if 'rsi' in technical_metrics:
                    rsi_val = technical_metrics['rsi']
                    analysis_parts.append(f"RSI: {rsi_val:.1f}")
                if 'atr' in technical_metrics:
                    atr_val = technical_metrics['atr']
                    analysis_parts.append(f"ATR: {atr_val:.5f}")
            
            # Add level information if available
            level_strength = signal.get('level_strength', None)
            if level_strength:
                analysis_parts.append(f"Level Strength: {level_strength}")
            
            # Join analysis parts
            enhanced_analysis = " | ".join(analysis_parts) if analysis_parts else "Price action signal"
            
            # Check for multiple TPs and format them if they exist
            take_profits = signal.get('take_profits')
            if take_profits and isinstance(take_profits, list):
                tp_text = ", ".join([f"TP{i+1}: {tp:.5f}" for i, tp in enumerate(take_profits)])
            else:
                tp_text = f"{take_profit:.5f}" # Fallback to single TP
            
            trade_details = (
                f"🔸 Strategy: {strategy_name}\n"
                f"🔹 Symbol: {symbol}\n"
                f"🔹 Direction: {(direction or '').upper()}\n"
                f"🔹 Entry: {entry_price:.5f}\n"
                f"🔹 Stop Loss: {stop_loss:.5f}\n"
                f"🔹 Take Profit: {tp_text}\n"
                f"🔹 Size: {position_size} lots\n\n"
                f"📊 Confidence: {confidence_pct:.1f}%\n"
                f"📊 Risk:Reward: {risk_reward_ratio:.2f}\n"
                f"📊 Pattern: {pattern}\n\n"
                f"📝 Analysis:\n{enhanced_analysis}"
            )
        
        return trade_details

    def _get_score_emoji(self, score: float) -> str:
        """
        Get a corresponding emoji based on the score percentage.
        
        Args:
            score: The score percentage
            
        Returns:
            str: The corresponding emoji
        """
        if score >= 80: return "⭐⭐⭐⭐⭐"
        elif score >= 60: return "⭐⭐⭐⭐"
        elif score >= 40: return "⭐⭐⭐"
        elif score >= 20: return "⭐⭐"
        else: return "⭐" 