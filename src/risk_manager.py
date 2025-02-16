from typing import Dict, Optional, List, Tuple
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta, UTC
import json

from config.config import TRADING_CONFIG
from src.models import Trade

class RiskManager:
    def __init__(self):
        # Core risk parameters
        self.max_risk_per_trade = 0.01  # 1% max risk per trade
        self.max_daily_loss = 0.015  # 1.5% max daily loss
        self.max_weekly_loss = 0.04  # 4% max weekly loss
        self.max_monthly_loss = 0.08  # 8% max monthly loss
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
        
        # Enhanced partial profit targets
        self.partial_tp_levels = [
            {'size': 0.5, 'ratio': 1.0},  # 50% at 1R
            {'size': 0.3, 'ratio': 1.5},  # 30% at 1.5R
            {'size': 0.2, 'ratio': 2.0}   # 20% at 2R
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

        # New: Market condition adjustments
        self.market_condition_adjustments = {
            'trending': 1.0,      # Full size in trending market
            'ranging': 0.75,      # 75% size in ranging market
            'choppy': 0.5,        # 50% size in choppy market
            'pre_event': 0.0      # No trading before major events
        }

        # New: Dynamic stop loss adjustments
        self.stop_loss_adjustments = {
            'atr_multiplier': 1.5,    # Base ATR multiplier
            'volatility_scale': True,  # Scale with volatility
            'min_distance': 0.0010,    # Minimum 10 pip stop
            'max_distance': 0.0050     # Maximum 50 pip stop
        }

        self.open_trades: List[Dict] = []
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, 
                                 entry_price: float, stop_loss_price: float, symbol: str) -> float:
        
        logger.debug(f"Calculating position size for {symbol}")
        logger.debug(f"Account balance: {account_balance}, Risk per trade: {risk_per_trade}, Entry price: {entry_price}, Stop loss price: {stop_loss_price}")
        
        # Validate inputs
        if account_balance <= 0:
            logger.error(f"Invalid account balance: {account_balance}")
            return 0.0
        
        if risk_per_trade <= 0 or risk_per_trade > 1:
            logger.error(f"Invalid risk per trade: {risk_per_trade}. Must be between 0 and 1.")
            return 0.0
        
        if entry_price <= 0:
            logger.error(f"Invalid entry price: {entry_price}")
            return 0.0
        
        if stop_loss_price <= 0:
            logger.error(f"Invalid stop loss price: {stop_loss_price}")
            return 0.0
        
        # Calculate risk amount
        account_risk = account_balance * risk_per_trade
        logger.debug(f"Account risk: {account_risk}")
        
        # Calculate trade risk
        trade_risk = abs(entry_price - stop_loss_price)
        logger.debug(f"Trade risk: {trade_risk}")
        
        # Get pip value for symbol
        pip_value = self._get_pip_value(symbol)
        logger.debug(f"Pip value for {symbol}: {pip_value}")
        
        if pip_value == 0:
            logger.error(f"Invalid pip value for {symbol}")
            return 0.0
        
        # Calculate position size
        position_size = (account_risk / trade_risk) * pip_value
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.2f}")
        
        return position_size
    
    def calculate_risk_amount(
        self,
        account_balance: float,
        risk_percentage: float
    ) -> float:
        """Calculate risk amount based on account balance and risk percentage."""
        try:
            risk_amount = account_balance * risk_percentage
            max_risk_amount = account_balance * TRADING_CONFIG.get('max_risk_per_trade', 0.03)
            return min(risk_amount, max_risk_amount)
        except Exception as e:
            logger.error(f"Error calculating risk amount: {str(e)}")
            return 0.0
    
    def calculate_daily_risk(
        self,
        account_balance: float,
        open_trades: List[Trade],
        pending_trades: List[Trade]
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
    
    def can_open_new_trade(self, current_trades: List[Trade]) -> bool:
        """Check if a new trade can be opened based on maximum concurrent trades limit."""
        try:
            return len(current_trades) < self.max_concurrent_trades
        except Exception as e:
            logger.error(f"Error checking if can open new trade: {str(e)}")
            return False
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        direction: str,
        entry_price: float,
        initial_stop: float,
        current_stop: float,
        atr: float,
        profit_level: float = 0.5  # Start trailing after 0.5R profit
    ) -> float:
        """
        Calculate trailing stop level based on price action and profit.
        
        Args:
            current_price: Current market price
            direction: Trade direction ('BUY' or 'SELL')
            entry_price: Original entry price
            initial_stop: Initial stop loss level
            current_stop: Current stop loss level
            atr: Current ATR value
            profit_level: Required profit level to start trailing (in R-multiples)
        
        Returns:
            float: New stop loss level
        """
        try:
            initial_risk = abs(entry_price - initial_stop)
            current_profit = (current_price - entry_price) if direction == "BUY" else (entry_price - current_price)
            r_multiple = current_profit / initial_risk

            # Only trail if we're in sufficient profit
            if r_multiple < profit_level:
                return current_stop

            # Calculate trail amounts based on profit level
            if r_multiple >= 2.0:
                trail_factor = 1.0  # Tight trail at 2R+
            elif r_multiple >= 1.5:
                trail_factor = 1.5  # Medium trail at 1.5R+
            elif r_multiple >= 1.0:
                trail_factor = 2.0  # Wider trail at 1R+
            else:
                trail_factor = 2.5  # Very wide trail below 1R

            trail_distance = atr * trail_factor

            if direction == "BUY":
                new_stop = current_price - trail_distance
                # If trade is in profit and new_stop is below entry, lock stop to breakeven
                if current_price > entry_price and new_stop < entry_price:
                    new_stop = entry_price
                return max(current_stop, new_stop)
            else:  # SELL
                new_stop = current_price + trail_distance
                # If trade is in profit and new_stop is above entry, lock stop to breakeven
                if current_price < entry_price and new_stop > entry_price:
                    new_stop = entry_price
                return min(current_stop, new_stop)

        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return current_stop
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        atr_multiplier: float = 2.0
    ) -> float:
        """Calculate stop loss based on ATR and recent swing levels."""
        try:
            # Get latest ATR value
            atr = df['atr'].iloc[-1]
            
            # Calculate ATR-based stop distance
            stop_distance = atr * atr_multiplier
            
            # Find recent swing levels
            lookback = 20  # Look back period for swings
            if signal_type == "BUY":
                # For buy orders, find recent low
                recent_low = df['low'].rolling(window=lookback).min().iloc[-1]
                # Use the larger of ATR-based stop or swing-based stop
                stop_distance_from_swing = entry_price - recent_low
                stop_distance = max(stop_distance, stop_distance_from_swing)
                stop_loss = entry_price - stop_distance
            else:  # SELL
                # For sell orders, find recent high
                recent_high = df['high'].rolling(window=lookback).max().iloc[-1]
                # Use the larger of ATR-based stop or swing-based stop
                stop_distance_from_swing = recent_high - entry_price
                stop_distance = max(stop_distance, stop_distance_from_swing)
                stop_loss = entry_price + stop_distance
            
            # Ensure minimum stop distance
            min_stop_distance = entry_price * 0.001  # Minimum 0.1% stop
            if signal_type == "BUY":
                stop_loss = min(entry_price - min_stop_distance, stop_loss)
            else:
                stop_loss = max(entry_price + min_stop_distance, stop_loss)
            
            return round(stop_loss, 5)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            # Default to 1% stop loss if calculation fails
            return entry_price * (0.99 if signal_type == "BUY" else 1.01)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit based on risk-reward ratio and recent price action."""
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
            
            # Ensure minimum reward-risk ratio
            min_rr = 1.5
            actual_rr = take_profit_diff / stop_loss_diff
            if actual_rr < min_rr:
                take_profit_diff = stop_loss_diff * min_rr
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
            current_date = datetime.now(UTC).date()
            total_risk = sum(
                trade.get("risk_amount", 0.0)
                for trade in self.open_trades
                if trade.get("entry_time", datetime.now(UTC)).date() == current_date
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

    def calculate_position_size(self, account_balance: float, risk_per_trade: float, 
                              stop_loss_pips: float, current_drawdown: float,
                              volatility_state: str, session: str) -> float:
        """Calculate position size based on multiple factors."""
        # Base position size calculation
        base_position = (account_balance * self.max_risk_per_trade * risk_per_trade) / stop_loss_pips
        
        # Apply drawdown scaling
        for drawdown_level, scale in sorted(self.drawdown_position_scale.items()):
            if current_drawdown >= drawdown_level:
                base_position *= scale
                
        # Apply volatility scaling
        base_position *= self.volatility_position_scale.get(volatility_state, 1.0)
        
        # Apply session-based scaling
        base_position *= self.session_risk_multipliers.get(session, 0.0)
        
        return base_position

    def check_trade_allowed(self, current_positions: int, daily_trades: int, 
                          weekly_trades: int, daily_loss: float, weekly_loss: float,
                          last_trade_time: datetime, consecutive_losses: int,
                          correlations: Dict[str, float]) -> Tuple[bool, str]:
        """Check if new trade is allowed based on all conditions."""
        if current_positions >= self.max_concurrent_trades:
            return False, "Max concurrent trades reached"
            
        if daily_trades >= self.max_daily_trades:
            return False, "Max daily trades reached"
            
        if weekly_trades >= self.max_weekly_trades:
            return False, "Max weekly trades reached"
            
        if daily_loss >= self.max_daily_loss:
            return False, "Daily loss limit reached"
            
        if weekly_loss >= self.max_weekly_loss:
            return False, "Weekly loss limit reached"
            
        if consecutive_losses >= self.consecutive_loss_limit:
            return False, "Consecutive loss limit reached"
            
        if any(corr >= self.max_correlation for corr in correlations.values()):
            return False, "Correlation limit exceeded"
            
        # Check trade spacing
        if last_trade_time and (datetime.now() - last_trade_time).hours < self.min_trades_spacing:
            return False, "Minimum trade spacing not met"
            
        return True, "Trade allowed"

    def apply_partial_profits(self, position_size: float, entry_price: float, 
                            stop_loss: float) -> List[Dict]:
        """Calculate partial profit targets."""
        r_value = abs(entry_price - stop_loss)  # 1R value
        targets = []
        
        for level in self.partial_tp_levels:
            target_price = entry_price + (r_value * level['ratio']) if entry_price > stop_loss else entry_price - (r_value * level['ratio'])
            targets.append({
                'size': position_size * level['size'],
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
        correlation: float
    ) -> float:
        """
        Calculate position size dynamically based on multiple factors:
        - Account balance and risk amount
        - Market volatility state
        - Trading session
        - Market conditions
        - Current drawdown
        - Correlation with other positions
        """
        try:
            # Base position size calculation
            price_difference = abs(entry_price - stop_loss)
            if price_difference == 0:
                logger.error("Invalid stop loss - same as entry price")
                return 0.0

            # Calculate pip value
            standard_lot = 100000
            pip_value = 0.0001 if not symbol.endswith('JPY') else 0.01
            pip_value_usd = pip_value * standard_lot
            pips_at_risk = price_difference / pip_value

            # Base position size
            base_position_size = risk_amount / (pips_at_risk * pip_value_usd)

            # 1. Volatility Adjustment
            volatility_multiplier = self.volatility_position_scale.get(volatility_state, 1.0)
            position_size = base_position_size * volatility_multiplier

            # 2. Session-based Adjustment
            session_multiplier = self.session_risk_multipliers.get(session, 1.0)
            position_size *= session_multiplier

            # 3. Market Condition Adjustment
            market_multiplier = self.market_condition_adjustments.get(market_condition, 1.0)
            position_size *= market_multiplier

            # 4. Correlation-based Adjustment
            if correlation >= self.correlation_limits['max_correlation']:
                position_size *= self.correlation_limits['high_correlation_scale']

            # 5. Account Balance Health Check
            # Calculate current drawdown
            initial_balance = TRADING_CONFIG.get('initial_balance', account_balance)
            current_drawdown = max(0, (initial_balance - account_balance) / initial_balance)
            
            # Apply drawdown-based position scaling
            for drawdown_level, scale in sorted(self.drawdown_position_scale.items()):
                if current_drawdown >= drawdown_level:
                    position_size *= scale

            # 6. Recovery Mode Check
            if self.recovery_mode['enabled'] and current_drawdown >= self.recovery_mode['threshold']:
                position_size *= self.recovery_mode['position_scale']

            # 7. Dynamic Risk Adjustment based on Recent Performance
            if hasattr(self, 'open_trades') and self.open_trades:
                recent_trades = self.open_trades[-5:]  # Look at last 5 trades
                win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)
                
                if win_rate < 0.4:  # Poor performance
                    position_size *= 0.5  # Reduce size by 50%
                elif win_rate > 0.7:  # Strong performance
                    position_size *= 1.2  # Increase size by 20%

            # 8. Volatility-based Maximum Position Size
            max_position_multiplier = {
                'extreme': 0.3,  # 30% of normal max size
                'high': 0.5,     # 50% of normal max size
                'normal': 1.0,   # Normal max size
                'low': 1.2       # 120% of normal max size
            }.get(volatility_state, 1.0)
            
            max_allowed_size = TRADING_CONFIG.get('max_position_size', 1.0) * max_position_multiplier

            # Ensure minimum and maximum position sizes
            position_size = max(TRADING_CONFIG.get('min_position_size', 0.01), 
                              min(position_size, max_allowed_size))

            # Round to 2 decimal places
            position_size = round(position_size, 2)

            logger.info(f"Dynamic position size calculation for {symbol}:")
            logger.info(f"Base position size: {base_position_size:.2f}")
            logger.info(f"Volatility multiplier: {volatility_multiplier}")
            logger.info(f"Session multiplier: {session_multiplier}")
            logger.info(f"Market condition multiplier: {market_multiplier}")
            logger.info(f"Final position size: {position_size:.2f} lots")

            return position_size

        except Exception as e:
            logger.error(f"Error calculating dynamic position size: {str(e)}")
            return TRADING_CONFIG.get('min_position_size', 0.01)

    def calculate_dynamic_stops(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        volatility_state: str
    ) -> Tuple[float, List[Dict]]:
        """Calculate dynamic stop loss and take profit levels based on volatility."""
        try:
            # Get current ATR value with validation
            if 'atr' not in df.columns or df['atr'].isna().all():
                raise ValueError("ATR values not available")
            current_atr = df['atr'].iloc[-1]
            
            # Enhanced ATR multiplier based on volatility state and recent price action
            volatility_multipliers = {
                'high': {
                    'sl': 2.5, 
                    'tp': [1.5, 2.5, 3.5],
                    'trail_start': 0.8,  # Start trailing earlier in high volatility
                    'trail_factors': [1.5, 1.2, 1.0]  # Tighter trailing in high volatility
                },
                'normal': {
                    'sl': 2.0, 
                    'tp': [1.2, 2.0, 3.0],
                    'trail_start': 1.0,  # Standard trailing start
                    'trail_factors': [2.0, 1.5, 1.2]  # Normal trailing
                },
                'low': {
                    'sl': 1.5, 
                    'tp': [1.0, 1.5, 2.0],
                    'trail_start': 1.2,  # Start trailing later in low volatility
                    'trail_factors': [2.5, 2.0, 1.5]  # Wider trailing in low volatility
                }
            }
            
            multiplier = volatility_multipliers.get(volatility_state, volatility_multipliers['normal'])
            
            # Calculate base stop distance with ATR-based scaling
            base_stop_distance = current_atr * multiplier['sl']
            
            # Find recent swing levels for additional context
            lookback = 20
            if direction == "BUY":
                recent_low = df['low'].rolling(window=lookback).min().iloc[-1]
                recent_high = df['high'].rolling(window=lookback).max().iloc[-1]
                swing_based_stop = entry_price - (entry_price - recent_low) * 1.1  # 110% of swing distance
                potential_target = recent_high + (recent_high - recent_low) * 0.5  # Project next swing high
            else:
                recent_high = df['high'].rolling(window=lookback).max().iloc[-1]
                recent_low = df['low'].rolling(window=lookback).min().iloc[-1]
                swing_based_stop = entry_price + (recent_high - entry_price) * 1.1
                potential_target = recent_low - (recent_high - recent_low) * 0.5  # Project next swing low
            
            # Use the wider of ATR-based or swing-based stop
            if direction == "BUY":
                atr_based_stop = entry_price - base_stop_distance
                stop_loss = min(atr_based_stop, swing_based_stop)  # Use the wider stop
            else:
                atr_based_stop = entry_price + base_stop_distance
                stop_loss = max(atr_based_stop, swing_based_stop)  # Use the wider stop
            
            # Apply minimum and maximum constraints
            min_distance = max(0.0008, current_atr * 1.2)  # At least 8 pips or 1.2 ATR
            max_distance = min(0.0030, current_atr * 4.0)  # At most 30 pips or 4 ATR
            
            actual_distance = abs(entry_price - stop_loss)
            if actual_distance < min_distance:
                stop_loss = entry_price - min_distance if direction == "BUY" else entry_price + min_distance
            elif actual_distance > max_distance:
                stop_loss = entry_price - max_distance if direction == "BUY" else entry_price + max_distance
            
            # Ensure stop loss is on the correct side of entry
            if direction == "BUY" and stop_loss >= entry_price:
                stop_loss = entry_price - min_distance
            elif direction == "SELL" and stop_loss <= entry_price:
                stop_loss = entry_price + min_distance
            
            # Calculate dynamic take profit levels based on market structure
            take_profits = []
            stop_distance = abs(entry_price - stop_loss)
            
            # Calculate volatility-adjusted position sizes
            if volatility_state == 'high':
                sizes = [0.5, 0.3, 0.2]  # More aggressive scaling in high volatility
            elif volatility_state == 'low':
                sizes = [0.4, 0.4, 0.2]  # More conservative in low volatility
            else:
                sizes = [0.4, 0.3, 0.3]  # Balanced scaling
            
            # Calculate take profit levels using both ATR and swing analysis
            for i, (tp_mult, size) in enumerate(zip(multiplier['tp'], sizes)):
                # Base TP distance on ATR
                atr_based_tp = stop_distance * tp_mult
                
                # Adjust based on recent swings for last target
                if i == len(multiplier['tp']) - 1:  # Last target
                    if direction == "BUY":
                        swing_tp = abs(potential_target - entry_price)
                        tp_distance = max(atr_based_tp, swing_tp)
                    else:
                        swing_tp = abs(entry_price - potential_target)
                        tp_distance = max(atr_based_tp, swing_tp)
                else:
                    tp_distance = atr_based_tp
                
                # Calculate TP price
                if direction == "BUY":
                    tp_price = entry_price + tp_distance
                else:
                    tp_price = entry_price - tp_distance
                
                take_profits.append({
                    'price': round(tp_price, 5),
                    'size': size,
                    'r_multiple': round(tp_distance / stop_distance, 2),
                    'trail_start': multiplier['trail_start'],  # When to start trailing
                    'trail_factor': multiplier['trail_factors'][i]  # How tight to trail
                })
            
            return round(stop_loss, 5), take_profits
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stops: {str(e)}")
            # More conservative fallback with dynamic distances
            fallback_distance = max(0.0008, min(0.0015, entry_price * 0.001))  # 8-15 pips based on price
            if direction == "BUY":
                fallback_stop = entry_price - fallback_distance
                fallback_tp = entry_price + (fallback_distance * 1.5)
            else:
                fallback_stop = entry_price + fallback_distance
                fallback_tp = entry_price - (fallback_distance * 1.5)
            
            # Multiple fallback targets
            return round(fallback_stop, 5), [
                {'price': round(fallback_tp, 5), 'size': 0.5, 'r_multiple': 1.5},
                {'price': round(entry_price + (fallback_distance * 2.0) if direction == "BUY" 
                              else entry_price - (fallback_distance * 2.0), 5), 'size': 0.3, 'r_multiple': 2.0},
                {'price': round(entry_price + (fallback_distance * 2.5) if direction == "BUY"
                              else entry_price - (fallback_distance * 2.5), 5), 'size': 0.2, 'r_multiple': 2.5}
            ]

    def _save_trade_details(self, trade: Dict) -> None:
        """Save trade details with proper handling of take_profit."""
        try:
            # Ensure trade has take_profit field
            if 'take_profit' not in trade:
                # Calculate take profit based on stop loss and a default R:R ratio
                stop_distance = abs(trade['entry_price'] - trade['stop_loss'])
                if trade['direction'] == 'BUY':
                    trade['take_profit'] = trade['entry_price'] + (stop_distance * 2)  # 1:2 R:R
                else:
                    trade['take_profit'] = trade['entry_price'] - (stop_distance * 2)  # 1:2 R:R
                
            # Save trade details
            self.open_trades.append(trade)
            
        except Exception as e:
            logger.error(f"Error saving trade details: {str(e)}")

    def should_adjust_stops(
        self,
        trade: Dict,
        current_price: float,
        current_atr: float
    ) -> Tuple[bool, float]:
        """
        Check if stops should be adjusted based on trailing conditions.
        
        Args:
            trade: Current trade information
            current_price: Current market price
            current_atr: Current ATR value
        
        Returns:
            Tuple[bool, float]: (Should adjust, New stop level)
        """
        try:
            # Get initial values
            entry_price = trade['entry_price']
            initial_stop = trade['initial_stop'] if 'initial_stop' in trade else trade['stop_loss']
            current_stop = trade['stop_loss']
            direction = trade['direction']
            
            # Calculate current profit
            initial_risk = abs(entry_price - initial_stop)
            current_profit = (current_price - entry_price) if direction == "BUY" else (entry_price - current_price)
            r_multiple = current_profit / initial_risk
            
            # Get trailing parameters from take profits
            if 'partial_take_profits' in trade and trade['partial_take_profits']:
                # Find the next active take profit level
                active_tp = None
                for tp in trade['partial_take_profits']:
                    if direction == "BUY" and current_price < tp['price']:
                        active_tp = tp
                        break
                    elif direction == "SELL" and current_price > tp['price']:
                        active_tp = tp
                        break
                
                if active_tp and r_multiple >= active_tp.get('trail_start', 1.0):
                    new_stop = self.calculate_trailing_stop(
                        current_price=current_price,
                        direction=direction,
                        entry_price=entry_price,
                        initial_stop=initial_stop,
                        current_stop=current_stop,
                        atr=current_atr,
                        profit_level=active_tp.get('trail_start', 1.0)
                    )
                    
                    if direction == "BUY" and new_stop > current_stop:
                        return True, round(new_stop, 5)
                    elif direction == "SELL" and new_stop < current_stop:
                        return True, round(new_stop, 5)
            
            # Fallback trailing stop adjustment if no partial_take_profits defined
            if not trade.get('partial_take_profits') or len(trade.get('partial_take_profits')) == 0:
                default_trail_start = 0.5
                if r_multiple >= default_trail_start:
                    new_stop = self.calculate_trailing_stop(
                        current_price=current_price,
                        direction=direction,
                        entry_price=entry_price,
                        initial_stop=initial_stop,
                        current_stop=current_stop,
                        atr=current_atr,
                        profit_level=default_trail_start
                    )
                    if (direction == "BUY" and new_stop > current_stop) or \
                       (direction == "SELL" and new_stop < current_stop):
                        return True, round(new_stop, 5)
            
            return False, current_stop
            
        except Exception as e:
            logger.error(f"Error checking stop adjustment: {str(e)}")
            return False, current_stop