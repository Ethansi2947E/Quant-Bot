import traceback
from typing import Dict, List, Any, Optional
from loguru import logger
import asyncio
import json

from src.risk_manager import RiskManager
from src.telegram.telegram_bot import TelegramBot
from src.mt5_handler import MT5Handler
from config.config import TELEGRAM_CONFIG

class SignalProcessor:
    """
    Handles signal processing and trade execution functionality.
    
    This class is responsible for:
    - Processing trading signals
    - Executing trades based on signals
    - Handling signals with existing positions
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
        self.telegram_bot = telegram_bot if telegram_bot is not None else TelegramBot.get_instance()
        self.config = config or {}
        
        # State tracking
        self.active_trades = {}
        self.min_confidence = self.config.get("min_confidence", 0.5)  # Default to 50% confidence
        self.allow_position_additions = self.config.get("allow_position_additions", True)  # Default to allowing position additions
        
    async def initialize(self, config=None):
        """Initialize the SignalProcessor with the given configuration."""
        if config:
            self.config = config
            # Update derived values
            self.min_confidence = self.config.get("min_confidence", 0.5)
            self.allow_position_additions = self.config.get("allow_position_additions", True)
            
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
            result["valid"] = False  # This is already a boolean
            result["reason"] = "Risk manager not available"
            return result
            
        # Get account info if not provided
        if not account_info:
            account_info = self.mt5_handler.get_account_info() if self.mt5_handler else {}
            
        account_balance = account_info.get("balance", 0)
        if account_balance <= 0:
            logger.warning("Invalid account balance for validation")
            result["valid"] = False  # This is already a boolean
            result["reason"] = "Invalid account balance"
            return result
            
        # Get open trades for context
        open_trades = []
        if self.mt5_handler:
            open_trades = self.mt5_handler.get_open_positions()
        
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
                    logger.debug(f"[{signal.get('symbol', 'Unknown')}] Position size adjusted by risk manager: {validation['adjusted_position_size']}")
                
                # Log if signal doesn't comply with risk rules
                if not result.get("valid", False):
                    logger.warning(f"[{signal.get('symbol', 'Unknown')}] Signal doesn't comply with risk rules: {validation.get('reason', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"[{signal.get('symbol', 'Unknown')}] Error validating trade: {str(e)}")
            result["valid"] = False
            result["reason"] = f"Validation error: {str(e)}"
            
        return result
    
    async def process_signals(self, signals: List[Dict]) -> None:
        """
        Process trading signals and execute trades if appropriate.
        
        Args:
            signals: List of signal dictionaries
        """
        if not signals:
            return
            
        if not self.mt5_handler:
            logger.warning("MT5Handler not set, cannot process signals")
            return
            
        logger.info(f"Processing {len(signals)} signals")
        
        execution_results = []
        
        for signal in signals:
            try:
                # Basic validation
                if not isinstance(signal, dict):
                    logger.warning(f"Invalid signal format: {signal}")
                    continue
                    
                # Extract key values
                symbol = signal.get("symbol")
                direction = signal.get("direction")
                confidence = signal.get("confidence", 0.0)
                signal_source = signal.get("source", "")
                
                if not symbol or not direction:
                    logger.warning(f"Missing required fields in signal: {signal}")
                    continue
                    
                # Skip confidence check for signal_generator1
                skip_confidence_check = signal_source == "signal_generator1" or "signal_generator1" in str(signal)
                
                # Check confidence threshold unless coming from signal_generator1
                if not skip_confidence_check and confidence < self.min_confidence:
                    logger.debug(f"Signal for {symbol} rejected: confidence {confidence} below threshold {self.min_confidence}")
                    continue
                    
                # Check if symbol is tradable using get_symbol_info instead of is_symbol_tradable
                symbol_info = self.mt5_handler.get_symbol_info(symbol)
                if not symbol_info or not hasattr(symbol_info, 'trade_mode') or symbol_info.trade_mode == 0:
                    logger.warning(f"Symbol {symbol} is not tradable, skipping signal")
                    continue
                    
                # Get existing positions for this symbol by filtering from all open positions
                all_positions = self.mt5_handler.get_open_positions()
                existing_positions = [p for p in all_positions if p.get("symbol") == symbol]
                
                if existing_positions:
                    # Handle signal with existing positions
                    result = await self.handle_signal_with_existing_positions(signal, existing_positions)
                    execution_results.append({"symbol": symbol, "direction": direction, "result": result})
                else:
                    # Execute new trade
                    result = await self.execute_trade_from_signal(signal)
                    execution_results.append({"symbol": symbol, "direction": direction, "result": result})
                    
            except Exception as e:
                logger.error(f"Error processing signal {signal}: {str(e)}")
                logger.error(traceback.format_exc())
                execution_results.append({"symbol": signal.get("symbol", "Unknown"), "direction": signal.get("direction", "Unknown"), "result": {"success": False, "error": str(e)}})
        
        # Log execution results summary
        successful_trades = [r for r in execution_results if r.get("result", {}).get("success", False)]
        failed_trades = [r for r in execution_results if not r.get("result", {}).get("success", False)]
        
        if successful_trades:
            logger.info(f"[EXECUTION] Successfully placed {len(successful_trades)} trades: {', '.join([f'{r['symbol']} {r['direction']}' for r in successful_trades])}")
        
        if failed_trades:
            logger.warning(f"[EXECUTION] Failed to place {len(failed_trades)} trades: {', '.join([f'{r['symbol']} {r['direction']}' for r in failed_trades])}")
            
        return execution_results
                
    async def execute_trade_from_signal(self, signal: Dict, is_addition: bool = False) -> Dict:
        """
        Execute a trade based on a given signal.
        
        Args:
            signal: Signal dictionary containing trade details
            is_addition: Whether this is adding to an existing position
            
        Returns:
            Dict: Result of the trade execution
        """
        if not self.mt5_handler or not self.risk_manager:
            logger.warning("MT5Handler or RiskManager not set, cannot execute trade")
            return {"success": False, "error": "MT5Handler or RiskManager not available"}
            
        try:
            # Extract key values
            symbol = signal.get("symbol")
            direction = signal.get("direction", "").lower()
            entry_price = signal.get("entry_price") or signal.get("entry")
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")
            timeframe = signal.get("timeframe", "H1")
            confidence = signal.get("confidence", 0.5)
            reason = signal.get("reason", "No reason provided")
            
            # Validate required fields
            if not all([symbol, direction, entry_price, stop_loss, take_profit]):
                logger.warning(f"Missing required trade parameters in signal: {signal}")
                return {"success": False, "error": "Missing required trade parameters"}
                
            # Get account info for position sizing
            account_info = self.mt5_handler.get_account_info()
            
            if not account_info:
                logger.error("Could not retrieve account info for position sizing")
                return {"success": False, "error": "Could not retrieve account info"}
            
            # Validate trade against risk management rules
            validation_result = self.validate_trade(signal, account_info)
            
            if not validation_result.get("valid", False):
                logger.warning(f"Trade validation failed: {validation_result.get('reason', 'Unknown reason')}")
                return {"success": False, "error": f"Trade validation failed: {validation_result.get('reason', 'Unknown reason')}"}
              # Use adjusted position size if provided by validation
            lot_size = signal.get("position_size", 0)
            if "adjusted_position_size" in validation_result:
                lot_size = validation_result["adjusted_position_size"]
                logger.info(f"Position size adjusted to {lot_size} based on risk validation")
            
            # # Calculate lot size if not already provided
            # if lot_size <= 0:
            #     # Get account info for position sizing calculation
            #     account_info = self.mt5_handler.get_account_info()
            #     account_balance = account_info.get('balance', 0)
                
            #     # Get necessary parameters from signal
            #     symbol = signal.get("symbol", "")
            #     direction = signal.get("direction", "").lower()
            #     entry_price = signal.get("entry_price") or signal.get("entry", 0.0)
            #     stop_loss = signal.get("stop_loss", 0.0)
            #     risk_percent = self.config.get("risk_per_trade", 1.0)
                
            #     lot_size = self.risk_manager.calculate_position_size(
            #         account_balance=account_balance,
            #         risk_per_trade=risk_percent,
            #         entry_price=entry_price,
            #         stop_loss_price=stop_loss,
            #         symbol=symbol
            #     )
            
            # if lot_size <= 0:
            #     logger.warning(f"Invalid lot size {lot_size} calculated for {symbol}, skipping trade")
            #     return {"success": False, "error": f"Invalid lot size {lot_size}"}
                


            # OVERRIDE: Use fixed lot size of 0.01 for all trades
            lot_size = 0.01
            logger.info(f"Using fixed lot size of {lot_size} as requested")
            
            # Use open_buy or open_sell instead of open_position
            result = None
            
            # Extract the generator name and strategy (setup_type) from the signal
            generator_name = signal.get("generator", "Unknown")
            # Extract just the number from the generator name if possible (e.g., SignalGenerator3 -> SG3)
            generator_num = ''.join(c for c in generator_name if c.isdigit())
            generator_short = f"SG{generator_num}" if generator_num else "SG"
            
            # Format the comment to be short and simple (avoid special characters)
            comment = f"{generator_short} {timeframe}"
            
            logger.info(f"[EXECUTION] Attempting to open {direction.upper()} position for {symbol} with lot size {lot_size}, stop_loss {stop_loss}, take_profit {take_profit}")
            
            if direction == "buy":
                # Ensure no None values are passed
                symbol_val = symbol if symbol is not None else ""
                lot_size_val = lot_size if lot_size is not None else 0.0
                stop_loss_val = stop_loss if stop_loss is not None else 0.0
                take_profit_val = take_profit if take_profit is not None else 0.0
                
                result_ticket = self.mt5_handler.open_buy(
                    symbol=symbol_val,
                    volume=lot_size_val,
                    stop_loss=stop_loss_val,
                    take_profit=take_profit_val,
                    comment=comment
                )
                if result_ticket:
                    result = {"retcode": 0, "order": result_ticket, "success": True}
                    logger.info(f"[EXECUTION_SUCCESS] BUY order placed for {symbol}, ticket #{result_ticket}")
                else:
                    result = {"retcode": -1, "comment": "Failed to open buy position", "success": False}
                    logger.error(f"[EXECUTION_FAILURE] Failed to open BUY position for {symbol}")
            else:  # direction == "sell"
                # Ensure no None values are passed
                symbol_val = symbol if symbol is not None else ""
                lot_size_val = lot_size if lot_size is not None else 0.0
                stop_loss_val = stop_loss if stop_loss is not None else 0.0
                take_profit_val = take_profit if take_profit is not None else 0.0
                
                result_ticket = self.mt5_handler.open_sell(
                    symbol=symbol_val,
                    volume=lot_size_val,
                    stop_loss=stop_loss_val,
                    take_profit=take_profit_val,
                    comment=comment
                )
                if result_ticket:
                    result = {"retcode": 0, "order": result_ticket, "success": True}
                    logger.info(f"[EXECUTION_SUCCESS] SELL order placed for {symbol}, ticket #{result_ticket}")
                else:
                    result = {"retcode": -1, "comment": "Failed to open sell position", "success": False}
                    logger.error(f"[EXECUTION_FAILURE] Failed to open SELL position for {symbol}")
            
            if result and result.get("retcode") == 0:
                # Trade executed successfully
                ticket = result.get("order", 0)
                
                # Send notification
                if self.telegram_bot:
                    # Use TELEGRAM_CONFIG from imported module instead of trying to access it as a local variable
                    # Get chat_id from config or use None as default
                    from config.config import TELEGRAM_CONFIG
                    default_chat_id = TELEGRAM_CONFIG.get("chat_id", None)
                    
                    # Check if telegram_bot is properly initialized, but don't try to initialize it again
                    if not hasattr(self.telegram_bot, 'is_running') or not self.telegram_bot.is_running:
                        logger.warning("Telegram bot not properly initialized for sending trade alert")
                        # Don't try to initialize here, just log the warning
                    else:
                        # Telegram bot is running, proceed with sending alert
                        # Ensure all parameters have the correct types
                        symbol_str = str(symbol) if symbol is not None else "Unknown"
                        direction_str = str(direction) if direction is not None else "unknown"
                        entry_float = float(entry_price) if entry_price is not None else 0.0
                        sl_float = float(stop_loss) if stop_loss is not None else 0.0
                        tp_float = float(take_profit) if take_profit is not None else 0.0
                        confidence_float = float(confidence) if confidence is not None else 0.5
                        # Get analysis from signal if available
                        reason_str = str(signal.get("analysis", "No reason provided"))
                        
                        try:
                            await self.telegram_bot.send_trade_alert(
                                chat_id=default_chat_id,
                                symbol=symbol_str,
                                direction=direction_str,
                                entry=entry_float,
                                sl=sl_float,
                                tp=tp_float,
                                confidence=confidence_float,
                                reason=reason_str
                            )
                        except Exception as e:
                            logger.error(f"Failed to send trade alert: {str(e)}")
                
                # TODO: Save trade to database
                return {"success": True, "ticket": ticket}
                
            else:
                error_code = result.get("retcode", -1) if result else -1
                error_message = result.get("comment", "Unknown error") if result else "No result returned"
                logger.error(f"Failed to open position on {symbol}: {error_code} - {error_message}")
                
                # Send error notification
                if self.telegram_bot:
                    # Ensure symbol is not None
                    symbol_val = str(symbol) if symbol is not None else "Unknown"
                    
                    # Check if telegram_bot is properly initialized, but don't try to initialize it
                    if not hasattr(self.telegram_bot, 'is_running') or not self.telegram_bot.is_running:
                        logger.warning("Telegram bot not properly initialized for sending error alert")
                        # Don't try to initialize here, just log the warning
                    else:
                        try:
                            await self.telegram_bot.send_trade_error_alert(
                                symbol=symbol_val,
                                error_type="Execution Error",
                                details=error_message
                            )
                        except Exception as e:
                            logger.error(f"Failed to send error alert: {str(e)}")
                
                return {"success": False, "error": error_message, "error_code": error_code}
                    
        except Exception as e:
            logger.error(f"Error executing trade from signal {signal}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
            
    async def handle_signal_with_existing_positions(self, signal: Dict, existing_positions: List[Dict]) -> Dict:
        """
        Process a signal when there are already open positions for the symbol.
        
        Args:
            signal: Signal dictionary containing trade details
            existing_positions: List of existing position dictionaries
            
        Returns:
            Dict: Result of the operation
        """
        symbol = signal.get("symbol")
        direction = signal.get("direction", "").lower()
        
        # Group positions by direction
        buy_positions = [p for p in existing_positions if p.get("type") == 0]  # MT5 type 0 = BUY
        sell_positions = [p for p in existing_positions if p.get("type") == 1]  # MT5 type 1 = SELL
        
        # Check if we have positions in the same direction as the signal
        same_direction_positions = buy_positions if direction == "buy" else sell_positions
        
        if same_direction_positions:
            # We already have positions in this direction
            if self.allow_position_additions:
                # Add to existing position if allowed
                logger.info(f"Adding to existing {direction} position for {symbol}")
                result = await self.execute_trade_from_signal(signal, is_addition=True)
                return result
            else:
                logger.info(f"Skipping signal for {symbol} - already have {direction} position and additions are disabled")
                return {"success": False, "error": "Position already exists in this direction and additions are disabled"}
        else:
            # We have positions but in the opposite direction
            opposite_positions = sell_positions if direction == "buy" else buy_positions
            
            if opposite_positions:
                logger.info(f"Signal conflicts with existing {len(opposite_positions)} opposite positions for {symbol}")
                # We could implement logic to close opposite positions here
                # For now, just log the conflict
                return {"success": False, "error": "Conflicting position exists in opposite direction"}
            
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