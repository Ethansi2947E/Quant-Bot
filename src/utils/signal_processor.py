import traceback
from typing import Dict, List, Any, Optional
from loguru import logger
import asyncio
import json
import time
import hashlib

from src.risk_manager import RiskManager
from src.telegram.telegram_bot import TelegramBot
from src.mt5_handler import MT5Handler
from config.config import TELEGRAM_CONFIG, TRADING_CONFIG

class SignalProcessor:
    """
    Handles signal processing and trade execution functionality.
    
    This class is responsible for:
    - Processing trading signals
    - Executing trades based on signals
    - Handling signals with existing positions
    - Validating signals against real-time MT5 data before execution
    """
    
    def __init__(self, mt5_handler=None, risk_manager=None, telegram_bot=None, config=None):
        """
        Initialize the SignalProcessor.
        
        Args:
            mt5_handler: MT5Handler instance for executing trades
            risk_manager: RiskManager instance for position sizing
            telegram_bot: TelegramBot instance for notifications
            config: Configuration dictionary
        """
        self.mt5_handler = mt5_handler if mt5_handler is not None else MT5Handler()
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()
        self.telegram_bot = telegram_bot if telegram_bot else TelegramBot.get_instance()
        self.config = config or {}
        
        # State tracking
        self.active_trades = {}
        self.min_confidence = self.config.get("min_confidence", 0.5)  # Default to 50% confidence
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
            self.min_confidence = self.config.get("min_confidence", 0.5)
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
                - adjusted_position_size (float): Adjusted position size if needed
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
                # Make sure we only assign a boolean to the valid key
                if "valid" in validation and isinstance(validation["valid"], bool):
                    result["valid"] = validation["valid"]
                else:
                    # If validation result doesn't have a valid boolean, assume it's not valid
                    result["valid"] = False
                
                # Copy other fields from validation
                for key, value in validation.items():
                    if key != "valid":  # We already handled the valid key
                        result[key] = value
                
                # If validation has adjusted position size, use it
                if result.get("valid", False) and "adjusted_position_size" in validation:
                    logger.debug(f"[{symbol}] Position size adjusted by risk manager: {validation['adjusted_position_size']}")
                
                # Log if signal doesn't comply with risk rules
                if not result.get("valid", False):
                    logger.warning(f"[{symbol}] Signal doesn't comply with risk rules: {validation.get('reason', 'Unknown')}")
            
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
            logger.debug(f"[TIMING] 🔍 Processing signal {i+1}/{len(signals)}: {symbol} {direction} at {signal_start_time:.6f}")
            
            try:
                # Skip signals without confidence
                confidence_check_start = time.time()
                confidence = signal.get("confidence", 0)
                confidence_threshold = 0.60  # Minimum 60% confidence
                
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
                                         (p.get("type") == 0 and direction.lower() == "buy") or
                                         (p.get("type") == 1 and direction.lower() == "sell")]
                    
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
                result = await self.execute_trade_from_signal(signal)
                
                if result.get("status") == "success":
                    signal["status"] = "executed"
                    signal["ticket"] = result.get("order")
                    logger.info(f"Trade executed successfully for {symbol} {direction}. Ticket: {result.get('order')}")
                    
                    # Store the processed signal to avoid duplicates
                    self._add_processed_signal(signal)
                else:
                    signal["status"] = "failed"
                    signal["error"] = result.get("message", "Unknown error")
                    error_code = result.get("code", "Unknown code")
                    logger.error(f"Trade execution failed for {symbol} {direction}: {error_code} - {signal['error']}")
                
                trade_execution_time = time.time() - trade_execution_start
                execution_status = "SUCCESS" if result.get("status") == "success" else "FAILED"
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
        Execute a trade based on a validated signal.
        """
        # Start timing
        execution_start = time.time()
        logger.debug(f"[TIMING] 💰 Starting trade execution for {signal['symbol']} at {execution_start}")
        
        # Enhanced debugging for position addition logic
        symbol = signal.get('symbol', 'Unknown')
        direction = signal.get('direction', 'Unknown')
        logger.warning(f"⚠️ TRADE EXECUTION: Symbol={symbol}, Direction={direction}, is_addition={is_addition}")
        
        # Get existing positions for this symbol
        # Double-check for existing positions to verify our logic
        all_positions = self.mt5_handler.get_open_positions() if self.mt5_handler else []
        symbol_positions = [p for p in all_positions if p.get("symbol") == symbol]
        same_direction_positions = [p for p in symbol_positions if 
                                  (p.get("type") == 0 and direction.lower() == "buy") or 
                                  (p.get("type") == 1 and direction.lower() == "sell")]
        
        # Log existing positions information
        logger.warning(f"⚠️ TRADE EXECUTION: Found {len(all_positions)} total positions")
        logger.warning(f"⚠️ TRADE EXECUTION: Found {len(symbol_positions)} positions for {symbol}")
        logger.warning(f"⚠️ TRADE EXECUTION: Found {len(same_direction_positions)} positions in same direction")
        
        # List position tickets
        if same_direction_positions:
            tickets = [p.get("ticket", "unknown") for p in same_direction_positions]
            logger.warning(f"⚠️ TRADE EXECUTION: Existing position tickets in same direction: {tickets}")
        
        # CRITICAL FIX: If this is an addition, always re-check the setting
        if is_addition:
            from config.config import TRADING_CONFIG
            self.allow_position_additions = TRADING_CONFIG.get("allow_position_additions", False)
            
            # Log if this is a position addition
            logger.warning(f"⚠️ POSITION ADDITION CHECK: Requested for {symbol} {direction} (allow_position_additions={self.allow_position_additions})")
            
            # Check if there really are positions in the same direction 
            # This provides an extra safety check
            if not same_direction_positions:
                logger.warning(f"⚠️ INCONSISTENCY DETECTED: Marked as addition but no existing positions found in same direction")
            
            # Safety check: verify allow_position_additions one more time
            if not self.allow_position_additions:
                logger.warning(f"❌ POSITION ADDITION BLOCKED: Attempt when additions are disabled! Skipping execution for {symbol}")
                return {
                    'status': 'error',
                    'message': 'Position additions are disabled in settings',
                    'code': 'ADDITIONS_DISABLED'
                }
        else:
            # Not marked as an addition, but check if it should be
            if same_direction_positions:
                logger.warning(f"⚠️ POTENTIAL ISSUE: Trade not marked as addition but {len(same_direction_positions)} positions exist in same direction")
                # FINAL FAILSAFE: If additions are disabled, block this trade
                if not self.allow_position_additions:
                    logger.warning(f"🚫 ADDITION FAILSAFE TRIGGERED: Blocking trade for {symbol} {direction} as it would create an additional position")
                    return {
                        'status': 'error',
                        'message': 'Blocked unmarked position addition - positions exist but additions disabled',
                        'code': 'UNMARKED_ADDITION_BLOCKED'
                    }
        
        try:
            # Basic parameter validation
            if 'symbol' not in signal or 'direction' not in signal:
                logger.error(f"Invalid signal format: {signal}")
                return {
                    'status': 'error',
                    'message': 'Invalid signal format'
                }
            param_validation_time = time.time() - execution_start
            logger.debug(f"[TIMING] 💰 Basic parameter validation took {param_validation_time:.4f}s")
            
            # Check MT5 connection
            if not self.mt5_handler.is_connected():
                logger.error("MT5 not connected for trade execution")
                return {
                    'status': 'error',
                    'message': 'MT5 not connected'
                }
            logger.debug("MT5 connection is active")
            connection_check_time = time.time() - execution_start - param_validation_time
            logger.debug(f"[TIMING] 💰 MT5 connection check took {connection_check_time:.4f}s")
            
            # Get symbol information
            symbol = signal.get('symbol')
            direction = signal.get('direction')

            # Check if we're using fixed lot size from config
            from config.config import TRADING_CONFIG
            use_fixed_lot_size = TRADING_CONFIG.get('use_fixed_lot_size', False)
            fixed_lot_size = TRADING_CONFIG.get('fixed_lot_size', 0.01)
            
            # Only retrieve account info if not using fixed lot size
            account_info = None
            account_info_time = 0
            
            if use_fixed_lot_size:
                logger.info(f"Using fixed lot size of {fixed_lot_size} from config (bypassing account info check)")
            else:
                # Attempt to get account info (with retries)
                account_info_start = time.time()
                max_attempts = 3
                attempt = 1
                
                while attempt <= max_attempts:
                    logger.debug(f"Attempting to retrieve account info (attempt {attempt}/{max_attempts})")
                    account_info = self.mt5_handler.get_account_info()
                    
                    # Check if account info is valid
                    if account_info and isinstance(account_info, dict) and 'balance' in account_info:
                        break
                    
                    logger.warning(f"Retrieved invalid account info on attempt {attempt}, retrying...")
                    
                    # If this is not the last attempt, reinitialize MT5 connection
                    if attempt < max_attempts:
                        logger.info(f"Reinitializing MT5 connection before account info retry #{attempt+1}...")
                        self.mt5_handler.initialize()
                        await asyncio.sleep(attempt * 2)  # Exponential backoff
                    
                    attempt += 1
                
                account_info_time = time.time() - account_info_start
                logger.debug(f"[TIMING] 💰 Account info retrieval took {account_info_time:.4f}s")
                
                if not account_info or 'balance' not in account_info:
                    logger.error("Could not retrieve valid account info for position sizing after multiple attempts")
                    # Only return error if fixed lot size is not being used
                    return {
                        'status': 'error',
                        'message': 'Could not retrieve account info, MT5 connection issues',
                        'code': 'Unknown code'
                    }

            # Continue with the rest of the method...
            # Extract signal parameters
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            # Default to current market price if no entry price specified
            if entry_price == 0:
                current_tick = self.mt5_handler.get_symbol_tick(symbol)
                if current_tick:
                    entry_price = current_tick.bid if direction.upper() == 'SELL' else current_tick.ask
            
            # Get position size (from signal or calculate)
            position_size = signal.get('position_size', 0)
            
            # If no position size in signal, calculate based on risk parameters
            if position_size <= 0:
                if use_fixed_lot_size:
                    position_size = fixed_lot_size
                    logger.info(f"Using fixed position size: {position_size} lots")
                else:
                    # Risk-based calculation using account info
                    account_balance = account_info.get('balance', 10000)  # Default if not available
                    
                    # Use RiskManager to calculate position size if available
                    if self.risk_manager:
                        position_size = self.risk_manager.calculate_position_size(
                            account_balance=account_balance,
                            risk_per_trade=1.0,  # 1% risk per trade
                            entry_price=entry_price,
                            stop_loss_price=stop_loss,
                            symbol=symbol
                        )
                    else:
                        # Basic calculation if no risk manager (fallback)
                        risk_amount = account_balance * 0.01  # 1% risk
                        stop_distance = abs(entry_price - stop_loss)
                        if stop_distance > 0:
                            position_size = risk_amount / stop_distance
                            position_size = min(position_size, 0.1)  # Cap at 0.1 lots
                        else:
                            position_size = 0.01  # Minimum position size
                            
                    logger.info(f"Calculated risk-based position size: {position_size} lots")
            
            # Ensure minimum position size
            position_size = max(position_size, 0.01)
            
            # Normalize volume according to symbol's volume_step
            position_size = self.mt5_handler.normalize_volume(symbol, position_size)
            
            # Execute the trade
            trade_params = {
                'symbol': symbol,
                'volume': position_size,
                'price': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'type': direction.upper(),
                'comment': signal.get('reason', 'Signal trade'),
                'position_id': signal.get('position_id', 0),  # For position modifications
                'is_addition': is_addition
            }
            
            # Add the validated signal to processed signals to prevent duplicates
            self._add_processed_signal(signal)
            
            # MT5Handler doesn't have place_order but has place_market_order
            order_result = self.mt5_handler.place_market_order(
                symbol=symbol,
                order_type=direction.upper(),
                volume=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=signal.get('reason', 'Signal trade')
            )
            
            # Log execution time
            execution_time = time.time() - execution_start
            logger.debug(f"[TIMING] 💰 Total trade execution took {execution_time:.4f}s")
            
            if order_result:
                # Success
                logger.info(f"✅ Trade executed: {symbol} {direction} {position_size} lots")
                
                # Send alert if configured
                if self.telegram_bot:
                    trade_details = (
                        f"🔹 Symbol: {symbol}\n"
                        f"🔹 Direction: {direction.upper()}\n"
                        f"🔹 Entry: {entry_price}\n"
                        f"🔹 Stop Loss: {stop_loss}\n"
                        f"🔹 Take Profit: {take_profit}\n"
                        f"🔹 Size: {position_size} lots\n"
                        f"🔹 Reason: {signal.get('reason', 'N/A')}"
                    )
                    await self.telegram_bot.send_message(f"✅ Trade Executed\n\n{trade_details}")
                
                return {
                    'status': 'success',
                    'order': order_result.get('ticket', 0),
                    'message': 'Trade executed successfully',
                    'time_taken': execution_time
                }
            else:
                # Failed
                error_message = 'Failed to place order'
                error_code = 'MT5 Error'
                
                logger.error(f"❌ Trade execution failed: {error_message} ({error_code})")
                
                # Send alert if configured
                if self.telegram_bot:
                    await self.telegram_bot.send_message(
                        f"❌ Trade Execution Failed\n\n"
                        f"Symbol: {symbol}\n"
                        f"Direction: {direction.upper()}\n"
                        f"Error: {error_message}\n"
                        f"Code: {error_code}"
                    )
                
                return {
                    'status': 'error',
                    'message': error_message,
                    'code': error_code,
                    'time_taken': execution_time
                }
                
        except Exception as e:
            logger.error(f"Exception during trade execution: {str(e)}")
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
            
    async def _notify_trade_action(self, message: str) -> None:
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
        Add a signal to the processed signals dictionary to prevent duplicates.
        
        Args:
            signal: The signal dictionary to add
        """
        signal_hash = self._generate_signal_hash(signal)
        self.processed_signals[signal_hash] = time.time()
        logger.debug(f"Added signal hash {signal_hash} to processed signals") 