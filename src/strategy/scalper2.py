"""
AlphaFusion Scalper Strategy

This strategy implements the "AlphaFusion Scalper," a hybrid model designed for
short-term momentum trading on low timeframes like M1 and M5. It combines two
primary entry models within a single trend-following framework:

1.  **Momentum Breakout:** Enters when price closes decisively outside a 34-period
    EMA channel, signaling a strong push in the direction of the trend.

2.  **Pullback to TAZ:** Enters when price pulls back into the "Traders Action Zone"
    (TAZ) between the 10 SMA and 30 EMA, and then shows a strong, quantifiable
    price rejection, indicating a likely continuation of the trend.

The strategy uses the 10 SMA / 30 EMA crossover and the slope of a 34 EMA to
define the dominant trend. Risk management is dynamic, utilizing the Average True
Range (ATR) to set adaptive stop-loss and take-profit levels that respond to
current market volatility.
"""

import pandas as pd
import pandas_ta as ta  # Using pandas-ta for clean and efficient indicator calculation
from loguru import logger
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import RISK_MANAGER_CONFIG
from src.risk_manager import RiskManager

# --- Timeframe-Specific Parameter Profiles for AlphaFusion Scalper ---
# These profiles adapt the strategy's sensitivity for different timeframes.
TIMEFRAME_PROFILES = {
    "M1": {
        "slope_period": 5, "rsi_buy_threshold": 55, "rsi_sell_threshold": 45,
        "atr_period": 14, "atr_sl_multiplier": 1.5
    },
    "M5": {
        "slope_period": 3, "rsi_buy_threshold": 55, "rsi_sell_threshold": 45,
        "atr_period": 14, "atr_sl_multiplier": 2.0
    },
    "M15": {
        "slope_period": 3, "rsi_buy_threshold": 60, "rsi_sell_threshold": 40,
        "atr_period": 14, "atr_sl_multiplier": 2.5
    },
}
# A default profile for M5 is a sensible fallback.
DEFAULT_PROFILE = TIMEFRAME_PROFILES["M1"]


class AlphaFusionScalper(SignalGenerator):
    """
    An implementation of the AlphaFusion Scalper strategy.

    - Inherits from SignalGenerator.
    - Utilizes a hybrid breakout and pullback model.
    - Integrates with the framework's RiskManager for position sizing and validation.
    """

    def __init__(self,
                 primary_timeframe: str = "M1",
                 risk_percent: float = 0.005,  # 0.5% risk per trade
                 min_risk_reward: float = 1.8,
                 **kwargs):
        """
        Initializes the AlphaFusion Scalper strategy.

        Args:
            primary_timeframe (str): The main timeframe for signal generation (e.g., "M1", "M5").
            risk_percent (float): The percentage of capital to risk per trade.
            min_risk_reward (float): The minimum required risk-to-reward ratio for a trade.
            **kwargs: Catches any additional keyword arguments.
        """
        # --- 1. Call the parent constructor ---
        super().__init__(**kwargs)

        # --- 2. Set Strategy Identity ---
        self.name = "AlphaFusionScalper"
        self.description = "Hybrid breakout and pullback scalping strategy using an EMA channel and TAZ."
        self.version = "1.1.0"

        # --- 3. Set Timeframe Parameters ---
        # This strategy primarily operates on a single timeframe.
        self.primary_timeframe = primary_timeframe

        # --- 4. Set Custom Strategy Parameters ---
        self.min_risk_reward = min_risk_reward
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)

        # --- 5. Load Timeframe-Specific Profile ---
        self.slope_period = None
        self.rsi_buy_threshold = None
        self.rsi_sell_threshold = None
        self.atr_period = None
        self.atr_sl_multiplier = None
        self._load_timeframe_profile()

        # --- 6. Initialize State Tracking ---
        self.processed_bars = {}

    def _load_timeframe_profile(self):
        """ Loads parameters from TIMEFRAME_PROFILES based on the primary timeframe. """
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        self.slope_period = profile['slope_period']
        self.rsi_buy_threshold = profile['rsi_buy_threshold']
        self.rsi_sell_threshold = profile['rsi_sell_threshold']
        self.atr_period = profile['atr_period']
        self.atr_sl_multiplier = profile['atr_sl_multiplier']

        logger.info(
            f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: "
            f"slope_period={self.slope_period}, rsi_buy={self.rsi_buy_threshold}, "
            f"atr_sl_mult={self.atr_sl_multiplier}"
        )

    @property
    def required_timeframes(self) -> List[str]:
        """ Specifies the timeframes this strategy needs. """
        return [self.primary_timeframe]

    async def initialize(self) -> bool:
        """ Pre-start setup for the strategy. """
        logger.info(f"🔌 Initializing {self.name} v{self.version}")
        # Could pre-load more data here if needed for indicator warm-up.
        return True

    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        **kwargs
    ) -> List[Dict]:
        """ The core method where trading signals are generated. """
        if market_data is None:
            market_data = {}
            
        signals = []
        rm = RiskManager.get_instance()

        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)

            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or len(primary_df) < 50:
                logger.debug(f"[{sym}] Missing or insufficient market data for {self.primary_timeframe}.")
                continue
            
            # --- Prevent Re-processing ---
            try:
                last_timestamp = str(primary_df.index[-1])
                bar_key = (sym, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_timestamp:
                    continue
                self.processed_bars[bar_key] = last_timestamp
            except IndexError:
                continue

            # --- 1. Calculate all indicators ---
            df = self._calculate_indicators(primary_df.copy())
            if df.empty:
                continue

            # --- 2. Check for entry signals ---
            direction, entry_type = self._find_entry_signal(df, sym)
            if direction == 'none':
                continue

            # --- 3. Assemble and Validate the Signal ---
            last_candle = df.iloc[-1]
            entry = last_candle['close']
            atr_val = last_candle[f'ATRr_{self.atr_period}']
            
            # This check prevents division by zero or nonsensical trades if ATR is flat.
            if atr_val <= 0:
                logger.warning(f"[{sym}] Invalid ATR value ({atr_val}). Skipping signal.")
                continue

            # Calculate SL/TP
            stop_loss_pips = self.atr_sl_multiplier * atr_val
            if direction == 'buy':
                stop_loss = entry - stop_loss_pips
                take_profit = entry + (stop_loss_pips * self.min_risk_reward)
            else:  # sell
                stop_loss = entry + stop_loss_pips
                take_profit = entry - (stop_loss_pips * self.min_risk_reward)
            
            logger.info(
                f"[{sym}] Potential {direction.upper()} signal found ({entry_type}). "
                f"Entry: {entry:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}"
            )
            # Structure the signal dictionary as required by the framework
            signal_details = {
                "symbol": sym,
                "direction": direction,
                "entry_price": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": 0.8,  # Base confidence for a valid signal
                "description": f"AlphaFusion {entry_type.capitalize()} signal.",
                "detailed_reasoning": [
                    f"Entry Type: {entry_type}",
                    f"Trend Confirmed: EMA_SLOPE > 0 for buys, < 0 for sells",
                    f"RSI Confirmation: {last_candle['RSI_14']:.2f}"
                ],
                "signal_timestamp": str(last_candle.name),
            }

            # --- 4. Validate with Risk Manager ---
            validation_result = rm.validate_and_size_trade(signal_details)
            if validation_result['is_valid']:
                logger.success(
                    f"✅ [{sym}] Valid {direction.upper()} signal via {entry_type}. "
                    f"Entry: {entry:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}"
                )
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals

    # ==============================================================================
    # --- AlphaFusion Strategy-Specific Helper Methods ---
    # ==============================================================================

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all necessary indicators for the AlphaFusion strategy.
        Uses the pandas_ta library for efficiency.
        """
        try:
            # Main Trend and Channel Indicators
            df.ta.sma(length=10, append=True)
            df.ta.ema(length=30, append=True)
            df.ta.ema(length=34, close='high', append=True, col_names=('EMA_34_HIGH',))
            df.ta.ema(length=34, close='low', append=True, col_names=('EMA_34_LOW',))
            df.ta.ema(length=34, close='close', append=True, col_names=('EMA_34_CLOSE',))

            # Confirmation and Risk Management Indicators
            df.ta.rsi(length=14, append=True)
            assert self.atr_period is not None
            df.ta.atr(length=self.atr_period, append=True) # ATR for risk management

            # Quantified Trend "Slope"
            assert self.slope_period is not None
            df['EMA_34_SLOPE'] = df['EMA_34_CLOSE'].diff(periods=self.slope_period)
            
            # Remove rows with NaN values created by indicators
            return df.dropna().reset_index()
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return pd.DataFrame()

    def _find_entry_signal(self, df: pd.DataFrame, symbol: str) -> Tuple[str, str]:
        """
        The core logic engine. Checks for breakout or pullback conditions and logs analysis.

        Args:
            df (pd.DataFrame): The DataFrame with indicators.
            symbol (str): The symbol being analyzed, for logging purposes.

        Returns:
            A tuple of (direction, entry_type), e.g., ('buy', 'breakout') or ('none', 'none').
        """
        if len(df) < 2:
            return 'none', 'none'

        last = df.iloc[-1]
        prev = df.iloc[-2]
        last_timestamp = str(df.index[-1])

        # --- 1. Evaluate All Conditions ---
        # Trend
        cond_uptrend = last['SMA_10'] > last['EMA_30'] and last['EMA_34_SLOPE'] > 0
        cond_downtrend = last['SMA_10'] < last['EMA_30'] and last['EMA_34_SLOPE'] < 0

        # RSI Momentum
        cond_rsi_buy = last['RSI_14'] > self.rsi_buy_threshold
        cond_rsi_sell = last['RSI_14'] < self.rsi_sell_threshold

        # Price Action Rejection
        cond_bull_rejection = (last['close'] - last['low']) / (last['high'] - last['low'] + 1e-9) > 0.7
        cond_bear_rejection = (last['high'] - last['close']) / (last['high'] - last['low'] + 1e-9) > 0.7
        
        # Entry Triggers
        cond_buy_breakout = prev['close'] <= prev['EMA_34_HIGH'] and last['close'] > last['EMA_34_HIGH']
        cond_sell_breakout = prev['close'] >= prev['EMA_34_LOW'] and last['close'] < last['EMA_34_LOW']
        
        cond_buy_taz = last['low'] <= last['SMA_10'] and last['low'] >= last['EMA_30']
        cond_sell_taz = last['high'] >= last['SMA_10'] and last['high'] <= last['EMA_30']
        
        cond_buy_pullback = cond_buy_taz and cond_bull_rejection
        cond_sell_pullback = cond_sell_taz and cond_bear_rejection

        # Final Candidate Logic
        is_long_breakout = cond_uptrend and cond_rsi_buy and cond_buy_breakout
        is_long_pullback = cond_uptrend and cond_rsi_buy and cond_buy_pullback
        is_short_breakout = cond_downtrend and cond_rsi_sell and cond_sell_breakout
        is_short_pullback = cond_downtrend and cond_rsi_sell and cond_sell_pullback
        
        is_long_candidate = is_long_breakout or is_long_pullback
        is_short_candidate = is_short_breakout or is_short_pullback

        # --- 2. Build Structured Log ---
        log_msg = f"[{symbol}] Condition Analysis for Bar {last_timestamp}:\n"
        log_msg += f"  - LONG -> {'CANDIDATE' if is_long_candidate else 'REJECTED'}\n"
        log_msg += f"    {'✅' if cond_uptrend else '❌'} [Trend]     Uptrend (SMA10 > EMA30 & Slope > 0)\n"
        log_msg += f"    {'✅' if cond_rsi_buy else '❌'} [Momentum]  RSI ({last['RSI_14']:.2f}) > Threshold ({self.rsi_buy_threshold})\n"
        log_msg += f"    {'✅' if cond_buy_breakout else '❌'} [Entry]     Breakout: Price crossed above EMA 34 High\n"
        log_msg += f"    {'✅' if cond_buy_pullback else '❌'} [Entry]     Pullback: In TAZ ({'✅' if cond_buy_taz else '❌'}) & Bull Rejection ({'✅' if cond_bull_rejection else '❌'})\n"
        
        log_msg += f"  - SHORT -> {'CANDIDATE' if is_short_candidate else 'REJECTED'}\n"
        log_msg += f"    {'✅' if cond_downtrend else '❌'} [Trend]     Downtrend (SMA10 < EMA30 & Slope < 0)\n"
        log_msg += f"    {'✅' if cond_rsi_sell else '❌'} [Momentum]  RSI ({last['RSI_14']:.2f}) < Threshold ({self.rsi_sell_threshold})\n"
        log_msg += f"    {'✅' if cond_sell_breakout else '❌'} [Entry]     Breakout: Price crossed below EMA 34 Low\n"
        log_msg += f"    {'✅' if cond_sell_pullback else '❌'} [Entry]     Pullback: In TAZ ({'✅' if cond_sell_taz else '❌'}) & Bear Rejection ({'✅' if cond_bear_rejection else '❌'})"

        logger.info(log_msg)

        # --- 3. Return Signal ---
        if is_long_breakout:
            return 'buy', 'breakout'
        if is_long_pullback:
            return 'buy', 'pullback'
        if is_short_breakout:
            return 'sell', 'breakout'
        if is_short_pullback:
            return 'sell', 'pullback'

        return 'none', 'none'