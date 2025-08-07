"""
Mean Reversion Scalper Strategy

A scalping strategy designed for M1/M5 timeframes that fades overextensions
by entering when price touches Bollinger Bands but only when market conditions
indicate low directional bias (ADX < 20).

Strategy Rules:
- Long Entry: Price touches lower Bollinger Band + candle closes in top 33% of range + ADX < 20
- Stop Loss: Signal candle low - (0.5 * ATR)
- Take Profit: Middle Bollinger Band
- Only trades when ADX indicates weak trend to avoid fading strong moves
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import RISK_MANAGER_CONFIG
from src.risk_manager import RiskManager

# --- Timeframe-Specific Parameter Profiles ---
TIMEFRAME_PROFILES = {
    "M1": {"lookback": 50, "bb_period": 20, "bb_std": 2.0, "atr_period": 14, "adx_period": 14},
    "M5": {"lookback": 100, "bb_period": 20, "bb_std": 2.0, "atr_period": 14, "adx_period": 14},
    "M15": {"lookback": 150, "bb_period": 20, "bb_std": 2.0, "atr_period": 14, "adx_period": 14},
}
DEFAULT_PROFILE = {"lookback": 100, "bb_period": 20, "bb_std": 2.0, "atr_period": 14, "adx_period": 14}

class MeanReversionScalper(SignalGenerator):
    """
    Mean Reversion Scalper Strategy
    
    Fades overextensions on lower timeframes when market lacks directional bias.
    Designed for quick scalping entries with tight risk management.
    """

    def __init__(self,
                 # --- Essential Timeframe Parameters ---
                 primary_timeframe: str = "M5",
                 
                 # --- Strategy-Specific Parameters ---
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 atr_period: int = 14,
                 adx_period: int = 14,
                 adx_threshold: float = 20.0,
                 reversal_threshold: float = 0.67,  # 67% for top 33% of candle range
                 atr_stop_multiplier: float = 0.5,
                 risk_percent: float = 0.01,
                 **kwargs):
        """
        Initialize the Mean Reversion Scalper strategy.

        Args:
            primary_timeframe (str): Main execution timeframe (M1/M5 recommended)
            bb_period (int): Bollinger Bands period
            bb_std (float): Bollinger Bands standard deviation
            atr_period (int): ATR calculation period
            adx_period (int): ADX calculation period
            adx_threshold (float): Maximum ADX value to allow trades (below = low trend strength)
            reversal_threshold (float): Minimum ratio for candle to close in top portion of range
            atr_stop_multiplier (float): Multiplier for ATR-based stop loss
            risk_percent (float): Risk percentage per trade
        """
        # Call parent constructor
        super().__init__(**kwargs)

        # Strategy identity
        self.name = "MeanReversionScalper"
        self.description = "Scalping strategy that fades overextensions when trend strength is low"
        self.version = "1.0.0"

        # Timeframe setup - only using primary timeframe (no higher timeframe needed)
        self.primary_timeframe = primary_timeframe

        # Strategy parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.reversal_threshold = reversal_threshold
        self.atr_stop_multiplier = atr_stop_multiplier

        # Risk management
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)

        # Load timeframe-specific profile
        self.lookback = None
        self._load_timeframe_profile()

        # State tracking
        self.processed_bars = {}

    def _load_timeframe_profile(self):
        """Load parameters from TIMEFRAME_PROFILES based on the primary timeframe."""
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        self.lookback = profile.get('lookback', DEFAULT_PROFILE['lookback'])
        
        # Override with profile values if not explicitly set
        self.bb_period = profile.get('bb_period', self.bb_period)
        self.bb_std = profile.get('bb_std', self.bb_std)
        self.atr_period = profile.get('atr_period', self.atr_period)
        self.adx_period = profile.get('adx_period', self.adx_period)

        logger.info(
            f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: "
            f"lookback={self.lookback}, bb_period={self.bb_period}, bb_std={self.bb_std}, "
            f"atr_period={self.atr_period}, adx_period={self.adx_period}"
        )

    @property
    def required_timeframes(self) -> List[str]:
        """
        This strategy only requires the primary timeframe since we're using ADX
        on the execution timeframe instead of H1.
        """
        return [self.primary_timeframe]

    async def initialize(self) -> bool:
        """Initialize the strategy."""
        logger.info(f"🔌 Initializing {self.name} for {self.primary_timeframe} timeframe")
        logger.info(f"📊 Strategy parameters: ADX threshold={self.adx_threshold}, BB period={self.bb_period}")
        return True

    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Generate mean reversion scalping signals.

        Args:
            market_data: Market data dictionary
            **kwargs: Additional context

        Returns:
            List of signal dictionaries
        """
        if market_data is None:
            market_data = {}

        signals = []
        total_symbols = len(market_data)
        
        logger.info(f"🚀 {self.name} v{self.version} starting analysis")
        logger.info(f"📊 Analyzing {total_symbols} symbol(s) on {self.primary_timeframe} timeframe")
        logger.info(f"⚙️ Strategy parameters: ADX threshold={self.adx_threshold}, BB period={self.bb_period}")
        
        # Get RiskManager instance
        rm = RiskManager.get_instance()
        balance = kwargs.get("balance", rm.daily_stats.get('starting_balance', 10000))
        logger.debug(f"💰 Account balance: {balance}")

        # Process each symbol
        processed_count = 0
        for sym, frames in market_data.items():
            processed_count += 1
            logger.trace(f"[{processed_count}/{total_symbols}] Processing {sym}...")
            # Get primary timeframe data
            primary_df = frames.get(self.primary_timeframe)

            # Validate data
            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty:
                logger.debug(f"[{sym}] Missing or empty market data for {self.primary_timeframe}")
                continue

            if len(primary_df) < max(self.bb_period, self.atr_period, self.adx_period) + 10:
                logger.debug(f"[{sym}] Insufficient data for calculations")
                continue

            # Check for duplicate processing
            last_timestamp = str(primary_df.index[-1])
            key = (sym, self.primary_timeframe)
            if key in self.processed_bars and self.processed_bars[key] == last_timestamp:
                logger.trace(f"[{sym}] Skipping - already processed bar {last_timestamp}")
                continue
            self.processed_bars[key] = last_timestamp

            logger.info(f"🔍 [{sym}] Analyzing new {self.primary_timeframe} bar at {last_timestamp}")
            
            # Step 1: Check ADX condition (trend strength filter)
            logger.info(f"📊 [{sym}] Step 1: Checking ADX trend strength condition...")
            if not self._check_adx_condition(primary_df):
                logger.info(f"❌ [{sym}] ADX condition FAILED - trend too strong for mean reversion")
                continue
            logger.info(f"✅ [{sym}] ADX condition PASSED - weak trend detected")

            # Step 2: Check for Bollinger Band overextension
            logger.info(f"📈 [{sym}] Step 2: Checking Bollinger Band overextension...")
            bb_signal = self._check_bollinger_overextension(primary_df)
            if not bb_signal['is_overextended']:
                logger.info(f"❌ [{sym}] BB overextension FAILED - price not touching lower band")
                logger.debug(f"   Current low: {bb_signal.get('current_low', 'N/A'):.5f}, Lower BB: {bb_signal.get('lower_bb', 'N/A'):.5f}")
                continue
            logger.info(f"✅ [{sym}] BB overextension PASSED - price touched lower Bollinger Band")
            logger.info(f"   📍 Current low: {bb_signal['current_low']:.5f}, Lower BB: {bb_signal['lower_bb']:.5f}")

            # Step 3: Check for reversal confirmation (candle closes in top 33% of range)
            logger.info(f"🕯️ [{sym}] Step 3: Checking reversal confirmation (candle close position)...")
            reversal_confirmed = self._check_reversal_confirmation(primary_df)
            if not reversal_confirmed['confirmed']:
                logger.info(f"❌ [{sym}] Reversal confirmation FAILED - candle didn't close in top 33%")
                logger.info(f"   📊 Close position in range: {reversal_confirmed.get('close_position', 0):.1%} (need >67%)")
                continue
            logger.info(f"✅ [{sym}] Reversal confirmation PASSED - strong rejection of lower levels")
            logger.info(f"   📊 Close position in range: {reversal_confirmed['close_position']:.1%}")

            # Step 4: Calculate trade parameters
            logger.info(f"🧮 [{sym}] Step 4: Calculating trade parameters...")
            trade_params = self._calculate_trade_parameters(primary_df, bb_signal)
            if not trade_params:
                logger.warning(f"❌ [{sym}] Failed to calculate valid trade parameters")
                continue
            
            logger.info(f"📋 [{sym}] Trade parameters calculated successfully:")
            logger.info(f"   💰 Entry: {trade_params['entry']:.5f}")
            logger.info(f"   🛑 Stop Loss: {trade_params['stop_loss']:.5f}")
            logger.info(f"   🎯 Take Profit: {trade_params['take_profit']:.5f}")
            logger.info(f"   ⚖️ Risk/Reward Ratio: {trade_params['risk_reward_ratio']:.2f}")
            logger.info(f"   📏 ATR: {trade_params['atr_value']:.5f}")

            # Step 5: Create signal dictionary
            logger.info(f"📝 [{sym}] Step 5: Creating signal dictionary...")
            signal_details = {
                # Essential fields
                "symbol": sym,
                "direction": "buy",  # Always long for this strategy
                "entry_price": trade_params['entry'],
                "stop_loss": trade_params['stop_loss'],
                "take_profit": trade_params['take_profit'],

                # Informational fields
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": 0.80,
                "description": "Mean reversion scalp on Bollinger Band overextension",
                "detailed_reasoning": [
                    f"ADX({self.adx_period}) = {bb_signal['adx_value']:.2f} < {self.adx_threshold}",
                    f"Price touched lower BB: {bb_signal['lower_bb']:.5f}",
                    f"Reversal confirmation: {reversal_confirmed['close_position']:.1%} of range",
                    f"Stop loss: {trade_params['stop_loss']:.5f}",
                    f"Take profit: {trade_params['take_profit']:.5f}"
                ],

                # Additional fields
                "pattern": "BB Mean Reversion",
                "signal_timestamp": str(primary_df.index[-1]),
            }
            
            logger.info(f"📋 [{sym}] Signal details created:")
            logger.info(f"   📈 Direction: {signal_details['direction'].upper()}")
            logger.info(f"   🎯 Confidence: {signal_details['confidence']:.0%}")
            logger.info(f"   📅 Timestamp: {signal_details['signal_timestamp']}")

            # Step 6: Validate with RiskManager
            logger.info(f"🔍 [{sym}] Step 6: Validating with RiskManager...")
            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.success(
                    f"🎉 [{sym}] SIGNAL ACCEPTED! Mean reversion trade validated by RiskManager"
                )
                logger.success(f"   💰 Entry: {trade_params['entry']:.5f}")
                logger.success(f"   🛑 Stop: {trade_params['stop_loss']:.5f}")
                logger.success(f"   🎯 Target: {trade_params['take_profit']:.5f}")
                logger.success(f"   ⚖️ R:R: {trade_params['risk_reward_ratio']:.2f}")
                final_trade_params = validation_result['final_trade_params']
                signals.append(final_trade_params)
            else:
                logger.warning(f"❌ [{sym}] Signal REJECTED by RiskManager: {validation_result['reason']}")
                logger.warning(f"   💔 All conditions were met but risk management prevented trade")

        # Summary logging
        logger.info(f"📈 {self.name} analysis complete!")
        logger.info(f"📊 Processed {total_symbols} symbols, generated {len(signals)} signal(s)")
        
        if signals:
            logger.success(f"🎯 Generated {len(signals)} mean reversion trade signal(s):")
            for i, signal in enumerate(signals, 1):
                logger.success(f"   {i}. {signal['symbol']} - Entry: {signal['entry_price']:.5f}")
        else:
            logger.info(f"🔍 No signals generated - waiting for suitable mean reversion setups")
            
        return signals

    # ==============================================================================
    # --- Technical Indicator Helper Methods ---
    # ==============================================================================

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands using pandas rolling functions.
        
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        close = df['close']
        rolling_mean = close.rolling(window=self.bb_period).mean()
        rolling_std = close.rolling(window=self.bb_period).std()
        
        upper_band = rolling_mean + (rolling_std * self.bb_std)
        lower_band = rolling_mean - (rolling_std * self.bb_std)
        
        return upper_band, rolling_mean, lower_band

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Returns:
            pd.Series: ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr

    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        Simplified implementation that returns reasonable values for trend strength measurement.
        
        Returns:
            pd.Series: ADX values (approximated using volatility and momentum)
        """
        # Use a simplified approach: measure volatility as a proxy for trend strength
        # Lower volatility relative to recent price movement suggests weak trend
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate price momentum (rate of change)
        price_change = close.pct_change(periods=self.adx_period).abs()
        
        # Calculate volatility using High-Low range
        hl_range = (high - low) / close
        avg_volatility = hl_range.rolling(window=self.adx_period).mean()
        
        # ADX proxy: normalized momentum relative to volatility
        # Higher values indicate stronger trend, lower values indicate consolidation
        adx_proxy = (price_change / (avg_volatility + 1e-10)) * 50  # Scale to typical ADX range
        
        # Cap values at reasonable ADX range (0-100)
        adx_proxy = adx_proxy.clip(0, 100)
        
        return adx_proxy

    def _check_adx_condition(self, df: pd.DataFrame) -> bool:
        """
        Check if ADX is below threshold indicating weak trend strength.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            bool: True if ADX condition is met
        """
        try:
            adx = self._calculate_adx(df)
            current_adx = adx.iloc[-1]
            
            if pd.isna(current_adx):
                logger.debug("ADX calculation returned NaN")
                return False
                
            is_weak_trend = current_adx < self.adx_threshold
            logger.debug(f"   📈 ADX({self.adx_period}) = {current_adx:.2f} (threshold: {self.adx_threshold})")
            if is_weak_trend:
                logger.debug(f"   ✅ Trend strength is WEAK - suitable for mean reversion")
            else:
                logger.debug(f"   ❌ Trend strength is STRONG - avoid mean reversion trades")
            
            return is_weak_trend
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return False

    def _check_bollinger_overextension(self, df: pd.DataFrame) -> Dict:
        """
        Check if price has touched or broken below the lower Bollinger Band.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            dict: Contains overextension info and Bollinger Band values
        """
        try:
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            
            # Get current values
            current_low = df['low'].iloc[-1]
            current_lower_bb = bb_lower.iloc[-1]
            current_middle_bb = bb_middle.iloc[-1]
            
            # Calculate ADX for logging
            adx = self._calculate_adx(df)
            current_adx = adx.iloc[-1]
            
            # Check if low touched or broke lower BB
            is_overextended = current_low <= current_lower_bb
            
            logger.debug(f"   📊 Bollinger Bands Analysis:")
            logger.debug(f"      Upper BB: {bb_upper.iloc[-1]:.5f}")
            logger.debug(f"      Middle BB: {current_middle_bb:.5f}")
            logger.debug(f"      Lower BB: {current_lower_bb:.5f}")
            logger.debug(f"      Current Low: {current_low:.5f}")
            logger.debug(f"      Overextension: {is_overextended}")
            logger.debug(f"   📊 ADX for context: {current_adx:.2f}")
            
            return {
                'is_overextended': is_overextended,
                'lower_bb': current_lower_bb,
                'middle_bb': current_middle_bb,
                'upper_bb': bb_upper.iloc[-1],
                'current_low': current_low,
                'adx_value': current_adx
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'is_overextended': False}

    def _check_reversal_confirmation(self, df: pd.DataFrame) -> Dict:
        """
        Check if the candle closes in the top 33% of its range (reversal confirmation).
        
        Args:
            df: Market data DataFrame
            
        Returns:
            dict: Confirmation details
        """
        try:
            last_candle = df.iloc[-1]
            
            # Calculate candle range and close position
            candle_high = last_candle['high']
            candle_low = last_candle['low']
            candle_close = last_candle['close']
            
            candle_range = candle_high - candle_low
            
            if candle_range == 0:
                return {'confirmed': False, 'close_position': 0.0}
            
            # Calculate where the close is in the candle range (0 = low, 1 = high)
            close_position = (candle_close - candle_low) / candle_range
            
            # Check if close is in top 33% (close_position > 0.67)
            confirmed = close_position > self.reversal_threshold
            
            logger.debug(f"   🕯️ Candle Analysis:")
            logger.debug(f"      High: {candle_high:.5f}")
            logger.debug(f"      Low: {candle_low:.5f}")
            logger.debug(f"      Close: {candle_close:.5f}")
            logger.debug(f"      Range: {candle_range:.5f}")
            logger.debug(f"      Close Position: {close_position:.1%} (need >{self.reversal_threshold:.1%})")
            logger.debug(f"      Reversal Confirmed: {confirmed}")
            
            return {
                'confirmed': confirmed,
                'close_position': close_position,
                'candle_range': candle_range
            }
            
        except Exception as e:
            logger.error(f"Error checking reversal confirmation: {e}")
            return {'confirmed': False}

    def _calculate_trade_parameters(self, df: pd.DataFrame, bb_signal: Dict) -> Optional[Dict]:
        """
        Calculate entry, stop loss, and take profit levels.
        
        Args:
            df: Market data DataFrame
            bb_signal: Bollinger Band signal information
            
        Returns:
            dict: Trade parameters or None if calculation fails
        """
        try:
            last_candle = df.iloc[-1]
            
            # Entry price (close of signal candle)
            entry_price = last_candle['close']
            
            # Calculate ATR for stop loss
            atr = self._calculate_atr(df)
            current_atr = atr.iloc[-1]
            
            if pd.isna(current_atr):
                logger.error("ATR calculation returned NaN")
                return None
            
            # Stop loss: Low of signal candle - (0.5 * ATR)
            signal_candle_low = last_candle['low']
            stop_loss = signal_candle_low - (self.atr_stop_multiplier * current_atr)
            
            # Take profit: Middle Bollinger Band
            take_profit = bb_signal['middle_bb']
            
            # Validate trade setup
            if stop_loss >= entry_price:
                logger.warning("Invalid stop loss - stop loss >= entry price")
                return None
                
            if take_profit <= entry_price:
                logger.warning("Invalid take profit - take profit <= entry price")
                return None
            
            # Calculate risk-reward ratio
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            logger.debug(f"   💰 Trade Parameter Calculation:")
            logger.debug(f"      Entry Price: {entry_price:.5f}")
            logger.debug(f"      Signal Candle Low: {signal_candle_low:.5f}")
            logger.debug(f"      ATR: {current_atr:.5f}")
            logger.debug(f"      Stop Loss: {stop_loss:.5f} (Low - {self.atr_stop_multiplier}×ATR)")
            logger.debug(f"      Take Profit: {take_profit:.5f} (Middle BB)")
            logger.debug(f"      Risk: {risk:.5f}")
            logger.debug(f"      Reward: {reward:.5f}")
            logger.debug(f"      R:R Ratio: {risk_reward_ratio:.2f}")
            
            return {
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'atr_value': current_atr
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade parameters: {e}")
            return None
