from typing import Dict, Optional, List, Tuple, Any, Callable
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta, UTC
import json
import MetaTrader5 as mt5
import math

from config.config import TRADING_CONFIG, TELEGRAM_CONFIG
from src.mt5_handler import MT5Handler

class RiskManager:
    def __init__(self, mt5_handler: Optional[MT5Handler] = None):
        """
        Initialize the RiskManager with risk parameters.
        
        Args:
            mt5_handler: Optional MT5Handler instance for MT5 operations
        """
        # Initialize MT5 handler
        self.mt5_handler = mt5_handler or MT5Handler()
        
        # Load settings from config
        self.use_fixed_lot_size = TRADING_CONFIG.get("use_fixed_lot_size", False)
        self.fixed_lot_size = TRADING_CONFIG.get("fixed_lot_size", 0.01)
        self.max_lot_size = TRADING_CONFIG.get("max_lot_size", 0.3)
        
        # Core risk parameters
        self.max_risk_per_trade = 0.01  # 1% max risk per trade
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_daily_risk = 0.03  # 3% max daily risk exposure
        self.max_weekly_loss = 0.05  # 5% max weekly loss
        self.max_monthly_loss = 0.10  # 10% max monthly loss
        self.max_drawdown_pause = 0.05  # Pause at 5% drawdown
        
        # Enhanced position management
        self.max_concurrent_trades = 1  # Single position at a time
        self.max_daily_trades = 2  # Reduced from 3
        self.max_weekly_trades = 8  # Reduced from 12
        self.min_trades_spacing = 4  # Increased from 2 hours
        
        # Enhanced drawdown controls
        self.consecutive_loss_limit = 2
        self.drawdown_position_scale = {
            0.02: 0.75,   # 75% size at 2% drawdown
            0.03: 0.50,   # 50% size at 3% drawdown
            0.04: 0.25,   # 25% size at 4% drawdown
            0.05: 0.0     # Stop trading at 5% drawdown
        }
        
        # Enhanced partial profit targets with more levels and closer targets
        self.partial_tp_levels = [
            {'size': 0.4, 'ratio': 0.5},  # 40% at 0.5R
            {'size': 0.3, 'ratio': 1.0},  # 30% at 1R
            {'size': 0.3, 'ratio': 1.5}   # 30% at 1.5R
        ]
        
        # Enhanced volatility-based position sizing
        self.volatility_position_scale = {
            'extreme': 0.25,  # 25% size in extreme volatility
            'high': 0.50,     # 50% size in high volatility
            'normal': 1.0,    # Normal size
            'low': 0.75       # 75% size in low volatility
        }
        
        # Enhanced recovery mode
        self.recovery_mode = {
            'enabled': True,
            'threshold': 0.05,        # 5% drawdown activates recovery
            'position_scale': 0.5,    # 50% position size
            'win_streak_required': 3,  # Need 3 winners to exit
            'max_trades_per_day': 2,   # Limited trades in recovery
            'min_win_rate': 0.40      # Min win rate to exit recovery
        }
        
        # Enhanced correlation controls
        self.correlation_limits = {
            'max_correlation': 0.7,    # Maximum allowed correlation
            'lookback_period': 20,     # Days for correlation calc
            'min_trades_for_calc': 50, # Minimum trades for reliable correlation
            'high_correlation_scale': 0.5  # Position scale for high correlation
        }
        
        # Enhanced session-based risk adjustments
        self.session_risk_multipliers = {
            'london_open': 1.0,    # Full size during London open
            'london_ny_overlap': 1.0,  # Full size during overlap
            'ny_open': 1.0,       # Full size during NY open
            'asian': 0.5,         # Half size during Asian session
            'pre_news': 0.0,      # No trading before high-impact news
            'post_news': 0.5      # Half size after news
        }

        # Market condition adjustments
        self.market_condition_adjustments = {
            'trending': 1.0,      # Full size in trending market
            'ranging': 0.75,      # 75% size in ranging market
            'choppy': 0.5,        # 50% size in choppy market
            'pre_event': 0.0      # No trading before major events
        }

        # Dynamic stop loss adjustments
        self.stop_loss_adjustments = {
            'atr_multiplier': 1.5,    # Base ATR multiplier
            'volatility_scale': True,  # Scale with volatility
            'min_distance': 0.0010,    # Minimum 10 pip stop
            'max_distance': 0.0050     # Maximum 50 pip stop
        }

        # Dynamic position sizing based on confidence
        self.confidence_position_scale = {
            0.90: 1.0,    # 100% size at 90%+ confidence
            0.80: 0.8,    # 80% size at 80-90% confidence
            0.70: 0.6,    # 60% size at 70-80% confidence
            0.60: 0.4,    # 40% size at 60-70% confidence
            0.50: 0.2     # 20% size at 50-60% confidence
        }
        
        # Dynamic take profit levels based on market conditions
        self.dynamic_tp_levels = {
            'trending': [
                {'size': 0.3, 'ratio': 1.0},  # 30% at 1R
                {'size': 0.4, 'ratio': 2.0},  # 40% at 2R
                {'size': 0.3, 'ratio': 3.0}   # 30% at 3R
            ],
            'ranging': [
                {'size': 0.5, 'ratio': 1.0},  # 50% at 1R
                {'size': 0.3, 'ratio': 1.5},  # 30% at 1.5R
                {'size': 0.2, 'ratio': 2.0}   # 20% at 2R
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

        self.open_trades: List[Dict] = []
    
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
            account_info = mt5.account_info()
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

    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float,
        symbol: str,
        market_condition: str = 'normal',
        volatility_state: str = 'normal',
        session: str = 'normal',
        correlation: float = 0.0,
        confidence_score: float = 0.5
    ) -> float:
        """
        Calculate position size based on multiple factors including risk, market conditions, and volatility.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade (0.0 to 1.0)
            entry_price: Trade entry price
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            market_condition: Market condition ('trending', 'ranging', 'choppy')
            volatility_state: Volatility state ('low', 'normal', 'high', 'extreme')
            session: Trading session ('london_open', 'ny_open', etc.)
            correlation: Correlation with other open positions (0.0 to 1.0)
            confidence_score: Trade setup confidence score (0.0 to 1.0)
            
        Returns:
            float: Calculated position size in lots
        """
        try:
            logger.debug(f"Calculating position size for {symbol}")
            
            # Safety limit for risk_per_trade - no more than 2%
            risk_per_trade = min(risk_per_trade, 0.02)
            
            # Check if using fixed lot size from config
            if self.use_fixed_lot_size:
                logger.info(f"Using fixed lot size from config: {self.fixed_lot_size} lots")
                
                # Get symbol info for volume constraints
                symbol_info = self.mt5_handler.get_symbol_info(symbol)
                if not symbol_info:
                    logger.error(f"Failed to get symbol info for {symbol}")
                    return self.fixed_lot_size
                
                # Get symbol-specific volume constraints
                min_lot = symbol_info.volume_min
                max_lot = min(symbol_info.volume_max, self.max_lot_size)
                lot_step = symbol_info.volume_step
                
                # Adjust fixed lot size to respect symbol constraints
                position_size = max(min_lot, min(self.fixed_lot_size, max_lot))
                position_size = round(position_size / lot_step) * lot_step
                
                logger.info(f"Final fixed position size for {symbol}: {position_size:.2f} lots")
                return position_size
        
            # Validate inputs
            if not self._validate_position_inputs(
                account_balance, risk_per_trade, entry_price, stop_loss_price):
                return 0.0
        
            # Get symbol info from MT5Handler
            symbol_info = self.mt5_handler.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                # Fallback calculation with more conservative measures
                logger.warning("Using fallback calculation with conservative measures")
                
                # Determine if this is a JPY pair
                is_jpy_pair = symbol.endswith('JPY') or 'JPY' in symbol
                
                # Calculate pip size and value based on symbol type
                if is_jpy_pair:
                    pip_size = 0.01
                    # For JPY pairs, contract size is typically 100,000
                    contract_size = 100000
                    # Calculate pip value for JPY pairs
                    pip_value = (pip_size / entry_price) * contract_size
                elif symbol.startswith('XAU'):
                    pip_size = 0.1
                    contract_size = 100
                    pip_value = pip_size * contract_size
                elif symbol.startswith('XAG'):
                    pip_size = 0.01
                    contract_size = 5000
                    pip_value = pip_size * contract_size
                # Special handling for cryptocurrency pairs
                elif symbol.endswith('USDm') or symbol.endswith('USDT') or symbol.endswith('USD') and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                    pip_size = 1.0
                    contract_size = 1
                    pip_value = pip_size
                else:
                    pip_size = 0.0001  # Default for most forex pairs
                    contract_size = 100000
                    pip_value = pip_size * contract_size
                
                # Calculate position size (more conservative approach)
                risk_amount = account_balance * risk_per_trade
                stop_distance_in_pips = abs(entry_price - stop_loss_price) / pip_size
                
                # Prevent division by zero
                if stop_distance_in_pips == 0:
                    logger.error("Stop distance is zero, cannot calculate position size")
                    return 0.0
                
                # Calculate position size in standard lots
                position_size = risk_amount / (stop_distance_in_pips * pip_value)
                
                # Apply a conservative cap (max 0.5% of account balance in lots)
                max_position = account_balance * 0.005
                position_size = min(position_size, max_position)
                
                # Minimum lot size
                min_lot = 0.01
                position_size = max(min_lot, position_size)
                
                # Always round down to 2 decimal places for safety
                position_size = math.floor(position_size * 100) / 100
                
                logger.info(f"Fallback position size calculation: {position_size:.2f} lots")
                return position_size
        
            # Get symbol-specific volume constraints
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            contract_size = symbol_info.trade_contract_size
            
            # Calculate base position size
            account_risk = account_balance * risk_per_trade
            trade_risk = abs(entry_price - stop_loss_price)
            
            # Get pip value specifically for the currency pair
            pip_value = self.calculate_pip_value(symbol, entry_price)
            
            # Calculate position size based on risk amount and stop distance
            if trade_risk == 0 or pip_value == 0:
                logger.error("Invalid trade risk or pip value")
                return 0.0
            
            # Calculate position size in standard lots
            pip_size = 0.0001
            if symbol.endswith('JPY') or 'JPY' in symbol:
                pip_size = 0.01
            stop_distance_in_pips = trade_risk / pip_size
            
            # Calculate base position size
            base_position = account_risk / (stop_distance_in_pips * pip_value)
            
            # Apply scaling factors
            position_size = base_position
            
            # 1. Market condition adjustment
            position_size *= self.market_condition_adjustments.get(market_condition, 1.0)
            
            # 2. Volatility adjustment
            position_size *= self.volatility_position_scale.get(volatility_state, 1.0)
            
            # 3. Session adjustment
            position_size *= self.session_risk_multipliers.get(session, 1.0)
            
            # 4. Correlation adjustment
            if correlation > self.correlation_limits['max_correlation']:
                position_size *= self.correlation_limits['high_correlation_scale']
            
            # 5. Confidence adjustment
            for threshold, scale in sorted(self.confidence_position_scale.items(), reverse=True):
                if confidence_score >= threshold:
                    position_size *= scale
                    break
            
            # 6. Drawdown adjustment
            current_drawdown = self.calculate_drawdown()
            for dd_level, scale in sorted(self.drawdown_position_scale.items()):
                if current_drawdown >= dd_level:
                    position_size *= scale
                    break
            
            # 7. Margin check - use a more conservative standard leverage
            standard_leverage = 100  # Standard leverage 1:100
            margin_required_per_lot = (entry_price * contract_size) / standard_leverage
            available_margin = account_balance * 0.3  # Use only 30% of balance for margin
            max_lots_by_margin = available_margin / margin_required_per_lot
            position_size = min(position_size, max_lots_by_margin)
            
            # 8. Apply account size-based cap (0.5% of account balance in lots)
            account_size_cap = account_balance * 0.005  # 0.5% cap
            position_size = min(position_size, account_size_cap)
            
            # 9. Final adjustments
            max_allowed_lot = min(self.max_lot_size, max_lot)  # Use max_lot_size from config
            position_size = math.floor(position_size / lot_step) * lot_step  # Round down to lot step
            position_size = max(min_lot, min(position_size, max_allowed_lot))
            
            logger.info(f"Final position size for {symbol}: {position_size:.2f} lots")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _validate_position_inputs(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float
    ) -> bool:
        """Validate inputs for position size calculation."""
        if account_balance <= 0:
            logger.error(f"Invalid account balance: {account_balance}")
            return False
        
        if risk_per_trade <= 0 or risk_per_trade > 1:
            logger.error(f"Invalid risk per trade: {risk_per_trade}")
            return False
        
        if entry_price <= 0:
            logger.error(f"Invalid entry price: {entry_price}")
            return False
        
        if stop_loss_price <= 0:
            logger.error(f"Invalid stop loss price: {stop_loss_price}")
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
            open_trades_risk = sum(
                abs(t.entry_price - t.stop_loss) * t.position_size * 100000
                for t in open_trades
                if t.timestamp.date() == current_date
            )
            
            # Calculate risk for pending trades
            pending_trades_risk = sum(
                abs(t.entry_price - t.stop_loss) * t.position_size * 100000
                for t in pending_trades
                if t.timestamp.date() == current_date
            )
            
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
        Calculate trailing stop level based on price action, profit, and market conditions.
        
        Args:
            trade: Dictionary containing trade information:
                - entry_price: Original entry price
                - initial_stop: Initial stop loss level
                - stop_loss: Current stop loss level
                - direction: Trade direction ('BUY' or 'SELL')
                - partial_take_profits: Optional list of take profit levels
            current_price: Current market price
            current_atr: Current ATR value (optional)
            market_condition: Current market condition ('trending', 'ranging', etc.)
        
        Returns:
            Tuple containing:
            - bool: Whether stop should be adjusted
            - float: New stop loss level
        """
        try:
            # Validate inputs
            if current_price == 0 or not trade.get('entry_price'):
                logger.error("Invalid price inputs for trailing stop calculation")
                return False, trade['stop_loss']

            # Get trade parameters
            entry_price = trade['entry_price']
            initial_stop = trade.get('initial_stop', trade['stop_loss'])
            current_stop = trade['stop_loss']
            direction = trade['direction']
            
            # Calculate current profit and R-multiple
            initial_risk = abs(entry_price - initial_stop)
            if initial_risk == 0:
                logger.warning("Initial risk is zero, cannot calculate R-multiple")
                return False, current_stop
            
            current_profit = (
                current_price - entry_price if direction == "BUY"
                else entry_price - current_price
            )
            r_multiple = current_profit / initial_risk
            
            # Determine profit level for trailing
            profit_level = 0.5  # Default to 0.5R for trailing
            
            # Check partial take profits for dynamic trailing
            if 'partial_take_profits' in trade and trade['partial_take_profits']:
                active_tp = None
                for tp in trade['partial_take_profits']:
                    if direction == "BUY" and current_price < tp['price']:
                        active_tp = tp
                        break
                    elif direction == "SELL" and current_price > tp['price']:
                        active_tp = tp
                        break
                
                if active_tp and r_multiple >= active_tp.get('trail_start', 0.3):
                    profit_level = active_tp.get('trail_start', 0.3)
            
            # Only trail if we have sufficient profit
            if r_multiple < profit_level:
                return False, current_stop
            
            # Calculate trail factor based on profit and market condition
            trail_factor = self.get_trail_factor(r_multiple, market_condition)
            
            # Calculate trail distance
            trail_distance = (
                current_atr * trail_factor if current_atr is not None
                else current_price * 0.001 * trail_factor  # Default to 0.1% * factor
            )
            
            # Calculate new stop level
            if direction == "BUY":
                new_stop = current_price - trail_distance
                # Lock in profits if beyond breakeven
                if current_price > entry_price and new_stop < entry_price:
                    new_stop = entry_price
                return True, max(current_stop, new_stop)
            
            else:  # SELL
                new_stop = current_price + trail_distance
                # Lock in profits if beyond breakeven
                if current_price < entry_price and new_stop > entry_price:
                    new_stop = entry_price
                return True, min(current_stop, new_stop)
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return False, trade['stop_loss']
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        atr_value: Optional[float] = None,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
        volatility_state: str = 'normal',
        market_condition: str = 'normal'
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Calculate stop loss and take profit levels based on multiple factors.
        
        Args:
            df: DataFrame with OHLC data
            signal_type: Trade direction ('BUY' or 'SELL')
            entry_price: Trade entry price
            atr_value: Current ATR value (optional)
            swing_low: Recent swing low price (optional)
            swing_high: Recent swing high price (optional)
            volatility_state: Current volatility state ('low', 'normal', 'high')
            market_condition: Current market condition ('trending', 'ranging', etc.)
            
        Returns:
            Tuple containing:
            - float: Calculated stop loss price
            - List[Dict]: Take profit levels with size and price
        """
        try:
            # 1. Calculate ATR-based stop distance
            if atr_value is None and 'atr' in df.columns:
                atr_value = df['atr'].iloc[-1]
            elif atr_value is None:
                atr_value = entry_price * 0.001  # Default to 0.1% of price
            
            # Base ATR multiplier from settings
            base_multiplier = self.stop_loss_adjustments['atr_multiplier']
            
            # 2. Adjust multiplier based on volatility
            if self.stop_loss_adjustments['volatility_scale']:
                if volatility_state == 'high':
                    base_multiplier *= 1.3  # Wider stops in high volatility
                elif volatility_state == 'low':
                    base_multiplier *= 0.75  # Tighter stops in low volatility
            
            # 3. Adjust for market condition
            if market_condition == 'trending':
                base_multiplier *= 1.2  # Wider stops in trends
            elif market_condition == 'choppy':
                base_multiplier *= 0.8  # Tighter stops in choppy conditions
            
            # Calculate base stop distance
            base_stop_distance = atr_value * base_multiplier
            
            # 4. Consider swing levels for stop placement
            if signal_type == "BUY":
                if swing_low is not None:
                    swing_based_stop = swing_low
                    swing_distance = entry_price - swing_based_stop
                    if self.stop_loss_adjustments['min_distance'] <= swing_distance <= self.stop_loss_adjustments['max_distance']:
                        stop_loss = swing_based_stop
                    else:
                        stop_loss = entry_price - base_stop_distance
                else:
                    stop_loss = entry_price - base_stop_distance
            else:  # SELL
                if swing_high is not None:
                    swing_based_stop = swing_high
                    swing_distance = swing_based_stop - entry_price
                    if self.stop_loss_adjustments['min_distance'] <= swing_distance <= self.stop_loss_adjustments['max_distance']:
                        stop_loss = swing_based_stop
                    else:
                        stop_loss = entry_price + base_stop_distance
                else:
                    stop_loss = entry_price + base_stop_distance
            
            # 5. Ensure stop distance is within limits
            min_distance = max(
                entry_price * 0.001,  # 0.1% minimum
                self.stop_loss_adjustments['min_distance']
            )
            max_distance = min(
                entry_price * 0.005,  # 0.5% maximum
                self.stop_loss_adjustments['max_distance']
            )
            
            if signal_type == "BUY":
                stop_loss = max(
                    entry_price - max_distance,
                    min(entry_price - min_distance, stop_loss)
                )
            else:
                stop_loss = min(
                    entry_price + max_distance,
                    max(entry_price + min_distance, stop_loss)
                )
            
            # 6. Calculate take profit levels
            take_profit_levels = []
            risk = abs(entry_price - stop_loss)
            
            # Use market condition specific TP levels
            tp_configs = self.dynamic_tp_levels.get(
                market_condition.lower(),
                self.dynamic_tp_levels['ranging']  # Default to ranging
            )
            
            for tp_config in tp_configs:
                if signal_type == "BUY":
                    tp_price = entry_price + (risk * tp_config["ratio"])
                else:  # SELL
                    tp_price = entry_price - (risk * tp_config["ratio"])
                
                take_profit_levels.append({
                    "price": round(tp_price, 5),
                    "size": tp_config["size"],
                    "ratio": tp_config["ratio"]
                })
            
            logger.info(f"Stop loss calculation for {signal_type}:")
            logger.info(f"  Entry: {entry_price:.5f}")
            logger.info(f"  Stop: {stop_loss:.5f} (Distance: {abs(entry_price - stop_loss):.5f})")
            logger.info(f"  ATR: {atr_value:.5f}, Multiplier: {base_multiplier:.2f}")
            
            return round(stop_loss, 5), take_profit_levels
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            # Return conservative default values
            default_stop = entry_price * (0.997 if signal_type == "BUY" else 1.003)
            default_tp = [
                {
                    "price": entry_price * (1.003 if signal_type == "BUY" else 0.997),
                    "size": 1.0,
                    "ratio": 1.0
                }
            ]
            return round(default_stop, 5), default_tp
    
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
        
        Args:
            entry_price: Trade entry price
            stop_loss: Stop loss price
            signal_type: Trade direction ('BUY' or 'SELL')
            market_condition: Market condition ('trending', 'ranging', etc.)
            risk_reward_ratio: Optional override for R:R ratio
            
        Returns:
            Tuple containing:
            - List of take profit levels with size and price
            - Final take profit price (weighted average)
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
            default_tp = (
                entry_price + (risk * 1.5) if signal_type == "BUY"
                else entry_price - (risk * 1.5)
            )
            return ([{'size': 1.0, 'price': round(default_tp, 5), 'r_multiple': 1.5}],
                    round(default_tp, 5))
    
    def validate_trade(
        self,
        account_balance: float,
        risk_amount: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_type: str,
        confidence: float,
        current_daily_risk: float = 0.0,
        current_weekly_risk: float = 0.0,
        daily_trades: int = 0,
        weekly_trades: int = 0,
        current_drawdown: float = 0.0,
        consecutive_losses: int = 0,
        last_trade_time: Optional[datetime] = None,
        correlations: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Comprehensive trade validation checking all risk parameters and trade setup.
        
        Args:
            account_balance: Current account balance
            risk_amount: Proposed risk amount for new trade
            entry_price: Trade entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_type: Trade direction ('BUY' or 'SELL')
            confidence: Trade setup confidence score (0.0 to 1.0)
            current_daily_risk: Current daily risk exposure
            current_weekly_risk: Current weekly risk exposure
            daily_trades: Number of trades taken today
            weekly_trades: Number of trades taken this week
            current_drawdown: Current drawdown percentage
            consecutive_losses: Number of consecutive losing trades
            last_trade_time: Time of last trade (optional)
            correlations: Dictionary of correlations with other positions (optional)
            
        Returns:
            Tuple containing:
            - bool: Whether trade is valid
            - str: Reason for rejection if not valid
        """
        try:
            # 1. Validate price levels
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
            
            # 2. Check confidence threshold
            if confidence < 0.5:
                return False, f"Low confidence signal: {confidence:.2f}"
            
            # 3. Check risk limits
            if risk_amount > account_balance * self.max_risk_per_trade:
                return False, f"Risk amount {risk_amount:.2f} exceeds max risk per trade"
            
            total_daily_risk = current_daily_risk + risk_amount
            if total_daily_risk > account_balance * self.max_daily_risk:
                return False, f"Total daily risk {total_daily_risk:.2f} would exceed limit"
            
            total_weekly_risk = current_weekly_risk + risk_amount
            if total_weekly_risk > account_balance * self.max_weekly_loss:
                return False, f"Total weekly risk {total_weekly_risk:.2f} would exceed limit"
            
            # 4. Check trade frequency limits
            if daily_trades >= self.max_daily_trades:
                return False, f"Daily trade limit {self.max_daily_trades} reached"
            
            if weekly_trades >= self.max_weekly_trades:
                return False, f"Weekly trade limit {self.max_weekly_trades} reached"
            
            # 5. Check drawdown limits
            if current_drawdown >= self.max_drawdown_pause:
                return False, f"Max drawdown {self.max_drawdown_pause*100}% reached"
            
            # 6. Check consecutive losses
            if consecutive_losses >= self.consecutive_loss_limit:
                return False, f"Consecutive loss limit {self.consecutive_loss_limit} reached"
            
            # 7. Check trade spacing
            if last_trade_time:
                time_since_last = datetime.now(UTC) - last_trade_time
                min_spacing = timedelta(hours=self.min_trades_spacing)
                if time_since_last < min_spacing:
                    return False, f"Minimum trade spacing {self.min_trades_spacing}h not met"
            
            # 8. Check correlations
            if correlations:
                for symbol, corr in correlations.items():
                    if abs(corr) > self.correlation_limits['max_correlation']:
                        return False, f"Correlation with {symbol} ({corr:.2f}) exceeds limit"
            
            # All checks passed
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
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
        
        Args:
            account_balance: Current account balance
            risk_amount: Proposed risk amount for new trade
            current_daily_risk: Current daily risk exposure
            current_weekly_risk: Current weekly risk exposure
            daily_trades: Number of trades taken today
            weekly_trades: Number of trades taken this week
            current_drawdown: Current drawdown percentage
            consecutive_losses: Number of consecutive losing trades
            last_trade_time: Time of last trade (optional)
            correlations: Dictionary of correlations with other positions (optional)
            
        Returns:
            Tuple containing:
            - bool: Whether trade is allowed
            - str: Reason for rejection if not allowed
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

    def apply_partial_profits(self, position_size: float, entry_price: float, 
                            stop_loss: float, min_lot: float = 0.01) -> List[Dict]:
        """Calculate partial profit targets and skip orders below the minimum lot size."""
        r_value = abs(entry_price - stop_loss)  # 1R value
        targets = []
        for level in self.partial_tp_levels:
            calculated_size = position_size * level['size']
            # Skip target if calculated size is below the minimum lot
            if calculated_size < min_lot:
                continue
            target_price = (entry_price + (r_value * level['ratio']) 
                            if entry_price > stop_loss 
                            else entry_price - (r_value * level['ratio']))
            targets.append({
                'size': calculated_size,
                'price': target_price,
                'r_multiple': level['ratio']
            })
        return targets

    def calculate_dynamic_position_size(
        self,
        account_balance: float,
        risk_amount: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        market_condition: str,
        volatility_state: str,
        session: str,
        correlation: float,
        confidence_score: float
    ) -> float:
        """Calculate position size dynamically based on multiple factors."""
        try:
            # Get symbol info for proper lot sizing
            symbol_info = self.mt5_handler.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            # Get symbol-specific volume constraints
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            # Limit risk amount to maximum 0.25% of balance for safety
            max_risk = account_balance * 0.0025
            risk_amount = min(risk_amount, max_risk)
            
            # Calculate pip value and stop distance
            pip_value = self.calculate_pip_value(symbol, entry_price)
            stop_distance = abs(entry_price - stop_loss)
            min_stop_distance = entry_price * 0.001  # Ensure a minimum stop distance
            stop_distance = max(stop_distance, min_stop_distance)
            
            if pip_value == 0 or stop_distance == 0:
                logger.error("Invalid pip value or stop distance")
                return 0.0
        
            # Calculate base position size
            base_position = risk_amount / (stop_distance / pip_value)
            
            # Apply scaling factors
            position_size = base_position * min(confidence_score, 0.7)
            position_size *= 0.7 if market_condition == 'ranging' else 1.0
            position_size *= 0.6 if volatility_state == 'high' else 1.0
            
            # Calculate margin requirement (using proper contract size and leverage)
            contract_size = 100000  # Standard lot size
            leverage = 100  # Standard leverage 1:100
            margin_required_per_lot = (entry_price * contract_size) / leverage
            
            # Calculate maximum lots based on available margin (using 50% of balance)
            available_margin = account_balance * 0.5
            max_lots_by_margin = available_margin / margin_required_per_lot
            
            # Ensure we don't exceed margin limits
            position_size = min(position_size, max_lots_by_margin)
            
            # Cap at 0.3 lots or symbol max lot, whichever is smaller
            max_allowed_lot = min(0.3, max_lot)
            
            # Round to valid lot step
            position_size = round(position_size / lot_step) * lot_step
            
            # Ensure position size is within symbol's limits
            position_size = max(min_lot, min(position_size, max_allowed_lot))
            
            # Log the calculation details
            logger.info(f"Position size calculation for {symbol}:")
            logger.info(f"  Risk={risk_amount:.2f}")
            logger.info(f"  Base={base_position:.2f}")
            logger.info(f"  Min Lot={min_lot:.2f}")
            logger.info(f"  Lot Step={lot_step:.2f}")
            logger.info(f"  Max Lots by Margin={max_lots_by_margin:.4f}")
            logger.info(f"  Final={position_size:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def get_confidence_multiplier(self, confidence_score: float) -> float:
        """Get position size multiplier based on confidence score."""
        for threshold, multiplier in sorted(self.confidence_position_scale.items(), reverse=True):
            if confidence_score >= threshold:
                return multiplier
        return 0.0  # Return 0 if confidence is too low

    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_stats = {
            'total_risk': 0.0,
            'realized_pnl': 0.0,
            'trade_count': 0,
            'starting_balance': 0.0,
            'last_reset': datetime.now(UTC).date()
        }

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