"""
AlphaFusion Scalper Strategy v2.0

This strategy is the unified and quantified implementation of three distinct but
related trading concepts: EMA Channel Breakouts, General Pullbacks, and the
Traders Action Zone (TAZ). It is designed for scalping on low timeframes (M1, M5).

The core logic is as follows:
1.  **Unified Trend Filter:** A trade is only considered if the trend is confirmed by
    both a 10/30 SMA/EMA cross AND a positive/negative slope on the 34 EMA.
2.  **Dual Entry Triggers:**
    a.  **Breakout:** Enters on a strong price close outside the 34 EMA High/Low channel.
    b.  **Pullback (TAZ):** Enters when price pulls back into the zone between the
        10 SMA and 30 EMA and prints a quantifiable price rejection candle.
3.  **Dynamic Risk Management:** All exits (Stop Loss and Take Profit) are calculated
    dynamically using the Average True Range (ATR) to adapt to market volatility.
"""

import pandas as pd
import pandas_ta as ta
from loguru import logger
from typing import Optional, List, Dict, Tuple

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import RISK_MANAGER_CONFIG
from src.risk_manager import RiskManager

# --- Timeframe-Specific Parameter Profiles for AlphaFusion Scalper ---
TIMEFRAME_PROFILES = {
    "M1": {
        "slope_period": 5, "rsi_buy_threshold": 55, "rsi_sell_threshold": 45,
        "atr_period": 20, "atr_sl_multiplier": 2.0, "min_risk_reward": 1.5
    },
    "M5": {
        "slope_period": 3, "rsi_buy_threshold": 55, "rsi_sell_threshold": 45,
        "atr_period": 14, "atr_sl_multiplier": 2.0, "min_risk_reward": 1.8
    },
}
DEFAULT_PROFILE = TIMEFRAME_PROFILES["M1"]


class AlphaFusionScalper2(SignalGenerator):
    """ A unified breakout and pullback scalping strategy. """

    def __init__(self,
                 primary_timeframe: str = "M1",
                 risk_percent: float = 0.005,  # 0.5% risk per trade
                 **kwargs):
        """ Initializes the AlphaFusion Scalper strategy. """
        super().__init__(**kwargs)

        self.name = "AlphaFusionScalper2"
        self.description = "Unified hybrid breakout and TAZ pullback strategy."
        self.version = "2.0.0"

        self.primary_timeframe = primary_timeframe
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)
        
        # Explicitly declare attributes loaded from profile for the type checker
        self.slope_period: int
        self.rsi_buy_threshold: int
        self.rsi_sell_threshold: int
        self.atr_period: int
        self.atr_sl_multiplier: float
        self.min_risk_reward: float

        self._load_timeframe_profile()
        self.processed_bars = {}

    def _load_timeframe_profile(self):
        """ Loads parameters from TIMEFRAME_PROFILES. """
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        for key, value in profile.items():
            setattr(self, key, value)
        logger.info(f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: {profile}")

    @property
    def required_timeframes(self) -> List[str]:
        return [self.primary_timeframe]

    async def initialize(self) -> bool:
        logger.info(f"🔌 Initializing {self.name} v{self.version}")
        return True

    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        **kwargs
    ) -> List[Dict]:
        if market_data is None: market_data = {}
            
        signals = []
        rm = RiskManager.get_instance()

        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)

            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or len(primary_df) < 50:
                continue
            
            last_timestamp = str(primary_df.index[-1])

            # --- Core Logic Pipeline ---
            logger.debug(f"[{sym}] Analyzing bar with timestamp: {last_timestamp}")
            df = self._calculate_indicators(primary_df.copy())
            if df.empty: continue

            direction, entry_type = self._find_entry_signal(df, sym, last_timestamp)
            if direction == 'none': continue

            # --- Assemble and Validate the Signal ---
            last_candle = df.iloc[-1]
            entry_price = last_candle['close']
            atr_val = last_candle[f'ATRr_{self.atr_period}']
            
            if atr_val <= 0: continue

            stop_loss_pips = self.atr_sl_multiplier * atr_val
            if direction == 'buy':
                stop_loss = entry_price - stop_loss_pips
                take_profit = entry_price + (stop_loss_pips * self.min_risk_reward)
            else:
                stop_loss = entry_price + stop_loss_pips
                take_profit = entry_price - (stop_loss_pips * self.min_risk_reward)
            
            signal_details = {
                "symbol": sym, "direction": direction, "entry_price": entry_price,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "timeframe": self.primary_timeframe, "strategy_name": self.name,
                "confidence": 0.85,
                "description": f"AlphaFusion {entry_type.capitalize()} signal.",
                "detailed_reasoning": [
                    f"Entry Type: {entry_type}",
                    f"Trend Confirmed: Both MA cross and EMA slope align",
                    f"RSI Confirmation: {last_candle['RSI_14']:.2f}"
                ],
                "signal_timestamp": str(last_candle.name),
            }

            validation_result = rm.validate_and_size_trade(signal_details)
            if validation_result['is_valid']:
                logger.success(f"✅ [{sym}] Valid {direction.upper()} signal via {entry_type}.")
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculates all indicators needed for the unified strategy. """
        try:
            # Trend Indicators (from all 3 strategies)
            df.ta.sma(length=10, append=True)
            df.ta.ema(length=30, append=True)
            df.ta.ema(length=34, close='close', append=True, col_names=('EMA_34_CLOSE',))
            # Breakout Channel (from Strategy 1)
            df.ta.ema(length=34, close='high', append=True, col_names=('EMA_34_HIGH',))
            df.ta.ema(length=34, close='low', append=True, col_names=('EMA_34_LOW',))
            # Confirmation Indicators (from all 3 strategies)
            df.ta.rsi(length=14, append=True)
            df.ta.sma(close='volume', length=20, append=True, col_names=('VOLUME_SMA_20',))
            # Risk Management
            df.ta.atr(length=self.atr_period, append=True)
            # Quantified Trend "Slope"
            df['EMA_34_SLOPE'] = df['EMA_34_CLOSE'].diff(periods=self.slope_period)
            return df.dropna().reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return pd.DataFrame()

    def _find_entry_signal(self, df: pd.DataFrame, symbol: str, last_timestamp: str) -> Tuple[str, str]:
        """ The core logic engine checking for breakout or pullback conditions. """
        if len(df) < 2: return 'none', 'none'

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # --- 1. Evaluate All Conditions ---
        # Trend
        cond_uptrend = last['SMA_10'] > last['EMA_30'] and last['EMA_34_SLOPE'] > 0
        cond_downtrend = last['SMA_10'] < last['EMA_30'] and last['EMA_34_SLOPE'] < 0

        # Confirmation
        cond_rsi_buy = last['RSI_14'] > self.rsi_buy_threshold
        cond_rsi_sell = last['RSI_14'] < self.rsi_sell_threshold
        cond_volume = last['volume'] > last['VOLUME_SMA_20']
        
        # Price Action
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
        is_long_breakout = cond_uptrend and cond_rsi_buy and cond_volume and cond_buy_breakout
        is_long_pullback = cond_uptrend and cond_rsi_buy and cond_volume and cond_buy_pullback
        is_short_breakout = cond_downtrend and cond_rsi_sell and cond_volume and cond_sell_breakout
        is_short_pullback = cond_downtrend and cond_rsi_sell and cond_volume and cond_sell_pullback
        
        is_long_candidate = is_long_breakout or is_long_pullback
        is_short_candidate = is_short_breakout or is_short_pullback

        # --- 2. Build Structured Log ---
        log_msg = f"[{symbol}] Condition Analysis for Bar {last_timestamp}:\n"
        log_msg += f"  - LONG -> {'CANDIDATE' if is_long_candidate else 'REJECTED'}\n"
        log_msg += f"    {'✅' if cond_uptrend else '❌'} [Trend]     Strong Uptrend Confirmed\n"
        log_msg += f"    {'✅' if cond_rsi_buy else '❌'} [Momentum]  RSI ({last['RSI_14']:.2f}) > Threshold ({self.rsi_buy_threshold})\n"
        log_msg += f"    {'✅' if cond_volume else '❌'} [Volume]    Volume > SMA(20)\n"
        log_msg += f"    {'✅' if cond_buy_breakout else '❌'} [Entry]     Breakout: Price crossed above EMA 34 High\n"
        log_msg += f"    {'✅' if cond_buy_pullback else '❌'} [Entry]     Pullback: In TAZ ({'✅' if cond_buy_taz else '❌'}) & Bull Rejection ({'✅' if cond_bull_rejection else '❌'})\n"
        
        log_msg += f"  - SHORT -> {'CANDIDATE' if is_short_candidate else 'REJECTED'}\n"
        log_msg += f"    {'✅' if cond_downtrend else '❌'} [Trend]     Strong Downtrend Confirmed\n"
        log_msg += f"    {'✅' if cond_rsi_sell else '❌'} [Momentum]  RSI ({last['RSI_14']:.2f}) < Threshold ({self.rsi_sell_threshold})\n"
        log_msg += f"    {'✅' if cond_volume else '❌'} [Volume]    Volume > SMA(20)\n"
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