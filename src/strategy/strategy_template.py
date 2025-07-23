"""
Strategy Template

This file serves as a template for creating new trading strategies within this framework.
It outlines the required structure, methods, and best practices to ensure compatibility
with the trading bot's orchestrator, data handling, and risk management systems.

To create a new strategy:
1.  Copy this file and rename it (e.g., `my_awesome_strategy.py`).
2.  Rename the class from `StrategyTemplate` to your strategy's name (e.g., `MyAwesomeStrategy`).
3.  Update the `__init__` method with your strategy's unique name, description, and parameters.
4.  Define your strategy's logic within the placeholder helper methods (_determine_trend, _find_key_levels, etc.).
5.  Flesh out the `generate_signals` method to connect your logic and produce valid signals.
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
# Import any technical analysis libraries you need, e.g., talib
import talib

# --- Best Practice: Timeframe-Specific Parameter Profiles ---
# Define different parameter sets for various timeframes to make your strategy adaptable.
TIMEFRAME_PROFILES = {
    "M5": {"lookback": 100, "ma_period": 20, "atr_multiplier": 2.0},
    "M15": {"lookback": 150, "ma_period": 50, "atr_multiplier": 2.5},
    "H1": {"lookback": 200, "ma_period": 100, "atr_multiplier": 3.0},
    # Add profiles for other timeframes your strategy might use
}
# A default profile is crucial as a fallback.
DEFAULT_PROFILE = {"lookback": 150, "ma_period": 50, "atr_multiplier": 2.5}

class StrategyTemplate(SignalGenerator):
    """
    A template for a new trading strategy.

    - Inherits from SignalGenerator.
    - Demonstrates parameter handling, multi-timeframe data usage, and risk management integration.
    """

    def __init__(self,
                 # --- Essential Timeframe Parameters ---
                 primary_timeframe: str = "M15",
                 higher_timeframe: str = "H1",

                 # --- Custom Strategy-Specific Parameters ---
                 # Add any parameters your strategy needs with sensible defaults.
                 ma_period: int = 50,
                 risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0,
                 **kwargs):
        """
        Initializes the strategy, setting up its core parameters.

        Args:
            primary_timeframe (str): The main timeframe for signal generation and execution.
            higher_timeframe (str): A longer timeframe used for trend analysis or context.
            ma_period (int): Example parameter for a moving average period.
            risk_percent (float): The percentage of capital to risk per trade.
            min_risk_reward (float): The minimum required risk-to-reward ratio for a trade.
            **kwargs: Catches any additional keyword arguments.
        """
        # --- 1. Call the parent constructor ---
        super().__init__(**kwargs)

        # --- 2. Set Strategy Identity ---
        # This is important for logging and tracking.
        self.name = "StrategyTemplate"
        self.description = "A template for creating new strategies."
        self.version = "1.0.0"

        # --- 3. Set Timeframe Parameters ---
        self.primary_timeframe = primary_timeframe
        self.higher_timeframe = higher_timeframe

        # --- 4. Set Custom Strategy Parameters ---
        self.ma_period = ma_period
        self.min_risk_reward = min_risk_reward

        # It's good practice to allow risk settings to be overridden by a central config.
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)

        # --- 5. Load Timeframe-Specific Profile ---
        # This section demonstrates how to use the TIMEFRAME_PROFILES dictionary.
        # It allows the strategy to adapt its parameters based on the chosen primary timeframe.
        self.lookback = None
        self.atr_multiplier = None
        self._load_timeframe_profile()

        # --- 6. Initialize State Tracking ---
        # Use dictionaries to keep track of processed bars or events to avoid duplicate signals.
        self.processed_bars = {}  # {(symbol, timeframe): last_processed_timestamp}


    def _load_timeframe_profile(self):
        """
        Loads parameters from TIMEFRAME_PROFILES based on the primary timeframe.
        This is a powerful pattern for creating adaptable strategies.
        """
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        self.lookback = profile.get('lookback', DEFAULT_PROFILE['lookback'])
        self.ma_period = profile.get('ma_period', self.ma_period) # Can be overridden by profile
        self.atr_multiplier = profile.get('atr_multiplier', DEFAULT_PROFILE['atr_multiplier'])

        logger.info(
            f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: "
            f"lookback={self.lookback}, ma_period={self.ma_period}, atr_multiplier={self.atr_multiplier}"
        )

    @property
    def required_timeframes(self) -> List[str]:
        """
        Specifies the timeframes this strategy needs. The bot orchestrator uses this
        to ensure the correct market data is passed to `generate_signals`.
        Using a set ensures there are no duplicates if timeframes are the same.
        """
        return list({self.higher_timeframe, self.primary_timeframe})

    async def initialize(self) -> bool:
        """
        An optional method to perform any setup required before the strategy starts.
        For example, pre-loading historical data or warming up indicators.
        """
        logger.info(f"🔌 Initializing {self.name}")
        return True

    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        symbol: Optional[str] = None, # This is usually handled by the loop below
        timeframe: Optional[str] = None, # This is also handled by the loop
        **kwargs
    ) -> List[Dict]:
        """
        The core method where trading signals are generated.

        Args:
            market_data (dict): A nested dictionary containing DataFrame objects for each
                                symbol and its required timeframes.
                                Format: {
                                    "SYMBOL": {
                                        "TIMEFRAME_1": pd.DataFrame,
                                        "TIMEFRAME_2": pd.DataFrame
                                    }
                                }
            **kwargs: Can contain additional context like account balance.

        Returns:
            list: A list of signal dictionaries. Each dictionary represents a potential trade.
        """
        if market_data is None:
            market_data = {}

        signals = []
        
        # --- How to use the RiskManager ---
        # Get the singleton instance of the RiskManager.
        rm = RiskManager.get_instance()
        # Get account balance if needed for calculations, with a fallback.
        balance = kwargs.get("balance", rm.daily_stats.get('starting_balance', 10000))

        # --- Loop Through Symbols Provided in market_data ---
        for sym, frames in market_data.items():
            # --- 1. Unpack DataFrames for Each Timeframe ---
            higher_df = frames.get(self.higher_timeframe)
            primary_df = frames.get(self.primary_timeframe)

            # --- 2. Validate Data ---
            # Always ensure you have valid DataFrame data before proceeding.
            if not isinstance(higher_df, pd.DataFrame) or higher_df.empty or \
               not isinstance(primary_df, pd.DataFrame) or primary_df.empty:
                logger.debug(f"[{sym}] Missing or empty market data for required timeframes.")
                continue

            last_timestamp = str(primary_df.index[-1])

            # --- Your Strategy's Logic Goes Here ---
            # Break down your logic into clear, testable steps using helper methods.

            # Step A: Determine the market trend using the higher timeframe.
            trend = self._determine_trend(higher_df)
            if trend == 'neutral':
                logger.debug(f"[{sym}] Trend is neutral, skipping.")
                continue

            # Step B: Identify key price levels (e.g., support/resistance) on the higher timeframe.
            key_levels = self._find_key_levels(higher_df)
            if not key_levels:
                logger.debug(f"[{sym}] No key levels found, skipping.")
                continue

            # Step C: On the primary timeframe, check if price is interacting with a key level.
            entry_condition_met, level_data = self._check_entry_condition(primary_df, key_levels, trend)
            if not entry_condition_met:
                continue

            # Step D: Look for a confirmation signal (e.g., candlestick pattern, indicator crossover).
            confirmation = self._find_confirmation_signal(primary_df)
            if not confirmation:
                logger.debug(f"[{sym}] No confirmation signal found.")
                continue

            # --- 4. Assemble the Signal Dictionary ---
            # If all conditions are met, calculate trade parameters and create the signal.
            
            # Example: Calculate Entry, Stop Loss, and Take Profit
            last_candle = primary_df.iloc[-1]
            entry = last_candle['close']
            atr = talib.ATR(primary_df['high'], primary_df['low'], primary_df['close'], timeperiod=14).iloc[-1]
            
            if trend == 'bullish':
                direction = 'buy'
                stop_loss = entry - (self.atr_multiplier * atr)
                take_profit = entry + (self.atr_multiplier * atr * self.min_risk_reward)
            else: # bearish
                direction = 'sell'
                stop_loss = entry + (self.atr_multiplier * atr)
                take_profit = entry - (self.atr_multiplier * atr * self.min_risk_reward)

            # --- How to Arrange a Signal ---
            # The signal dictionary must be structured correctly for the RiskManager.
            signal_details = {
                # --- Essential Fields ---
                "symbol": sym,
                "direction": direction, # 'buy' or 'sell'
                "entry_price": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,

                # --- Informational Fields ---
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": 0.75,  # A score from 0.0 to 1.0 indicating signal strength
                "description": f"Signal based on {trend} trend and confirmation pattern.",
                "detailed_reasoning": [
                    f"HTF Trend: {trend}",
                    f"Interaction with level: {level_data.get('price'):.5f}",
                    f"Confirmation: {confirmation.get('pattern_name')}"
                ],
                
                # --- Optional but Recommended Fields ---
                "pattern": confirmation.get('pattern_name', 'N/A'),
                "signal_timestamp": str(last_candle.name),
            }

            # --- 5. Validate the Trade with the RiskManager ---
            # This step is CRUCIAL. The RiskManager checks against account-level rules
            # (e.g., max exposure, daily loss limits) and calculates the correct position size.
            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.info(f"✅ [{sym}] Valid signal generated. Direction: {direction}, Entry: {entry:.5f}")
                # Use the final parameters returned by the RiskManager.
                final_trade_params = validation_result['final_trade_params']
                signals.append(final_trade_params)
            else:
                # Log why the trade was rejected.
                logger.warning(f"Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals

    # ==============================================================================
    # --- Placeholder Helper Methods ---
    # Replace the logic in these methods with your own strategy's implementation.
    # ==============================================================================

    def _determine_trend(self, df: pd.DataFrame) -> str:
        """
        Analyzes the provided DataFrame to determine the market trend.
        This is a placeholder. Implement your trend-defining logic here.

        Args:
            df (pd.DataFrame): Market data (typically from the higher timeframe).

        Returns:
            str: 'bullish', 'bearish', or 'neutral'.
        """
        logger.debug("Executing placeholder trend analysis...")
        # Example: Use a simple moving average crossover.
        try:
            ma_fast = talib.SMA(df['close'], timeperiod=21)
            ma_slow = talib.SMA(df['close'], timeperiod=self.ma_period)
            if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                return 'bullish'
            else:
                return 'bearish'
        except Exception:
            return 'neutral'

    def _find_key_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identifies significant support and resistance levels.
        This is a placeholder. Implement your S/R detection logic here.

        Args:
            df (pd.DataFrame): Market data.

        Returns:
            list: A list of dictionaries, where each dict represents a key level.
                  Example: [{'price': 1.2345, 'type': 'resistance'}]
        """
        logger.debug("Executing placeholder S/R level detection...")
        # Example: Find the last major swing high/low.
        lookback_period = self.lookback // 2
        last_swing_high = df['high'].iloc[-lookback_period:].max()
        last_swing_low = df['low'].iloc[-lookback_period:].min()
        return [
            {'price': last_swing_high, 'type': 'resistance'},
            {'price': last_swing_low, 'type': 'support'}
        ]

    def _check_entry_condition(self, df: pd.DataFrame, levels: List[Dict], trend: str) -> Tuple[bool, Dict]:
        """
        Checks if the price is interacting with a key level in a way that
        presents a potential trade setup.
        This is a placeholder.

        Args:
            df (pd.DataFrame): Market data (typically from the primary timeframe).
            levels (list): The key levels identified earlier.
            trend (str): The current market trend.

        Returns:
            tuple: (bool, dict) - A boolean indicating if the condition is met,
                   and the level data that triggered the condition.
        """
        logger.debug("Executing placeholder entry condition check...")
        last_close = df['close'].iloc[-1]
        tolerance = df['close'].mean() * 0.001 # 0.1% tolerance

        target_level_type = 'support' if trend == 'bullish' else 'resistance'

        for level in levels:
            if level['type'] == target_level_type:
                if abs(last_close - level['price']) <= tolerance:
                    return True, level # Condition met, return the level we are interacting with
        return False, {}

    def _find_confirmation_signal(self, df: pd.DataFrame) -> Dict:
        """
        Looks for a specific pattern or indicator signal to confirm the trade setup.
        This is a placeholder. Implement your confirmation logic here.

        Args:
            df (pd.DataFrame): Market data (primary timeframe).

        Returns:
            dict: A dictionary containing details of the confirmation, or an empty dict if none found.
                  Example: {'pattern_name': 'Bullish Engulfing', 'candle_time': '...'}
        """
        logger.debug("Executing placeholder confirmation signal search...")
        # Example: Check for a basic pin bar (long wick).
        last_candle = df.iloc[-1]
        body = abs(last_candle['open'] - last_candle['close'])
        wick_low = last_candle['low'] - min(last_candle['open'], last_candle['close'])

        if body > 0 and wick_low / body > 2.0: # Lower wick is >2x body size
             return {'pattern_name': 'Pin Bar (Bullish)', 'timestamp': str(last_candle.name)}
        return {} 