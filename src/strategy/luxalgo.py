"""
Lux Algo Premium Strategy

This strategy is a Python implementation of the core logic from the "Premium Lux Algo"
TradingView indicator. It focuses on the primary signal generation and the unique
"Smart Trail" exit mechanism.

Core Logic:
- Entry Signal: A crossover/crossunder of the price and a Supertrend line, confirmed
  by a simple moving average.
- Trend Filter: Uses a smoothed Heikin-Ashi indicator on a higher timeframe to
  ensure trades are taken in the direction of the major trend.
- Exit Logic: Implements the "Smart Trail", a sophisticated trailing stop-loss based
  on a Wilder's Moving Average of the True Range. Take-profit is calculated based
  on a fixed risk-to-reward ratio.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, List, Dict, Tuple

# Use pandas-ta for a wider range of indicators that map well to TradingView
import pandas_ta as ta


# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import get_risk_manager_config
from src.risk_manager import RiskManager

# --- Timeframe-Specific Parameter Profiles ---
# These profiles adjust the strategy's sensitivity and responsiveness for different timeframes.
# Lower timeframes (e.g., M5) use a higher sensitivity (factor) to filter out more noise.
TIMEFRAME_PROFILES = {
    "M5": {"supertrend_factor": 6.0, "smart_trail_atr_period": 14, "smart_trail_atr_factor": 4.5},
    "M15": {"supertrend_factor": 5.5, "smart_trail_atr_period": 13, "smart_trail_atr_factor": 4.0},
    "H1": {"supertrend_factor": 5.0, "smart_trail_atr_period": 12, "smart_trail_atr_factor": 3.5},
    "H4": {"supertrend_factor": 4.5, "smart_trail_atr_period": 11, "smart_trail_atr_factor": 3.0},
}
DEFAULT_PROFILE = TIMEFRAME_PROFILES["M15"]


class LuxAlgoPremiumStrategy(SignalGenerator):
    """
    A Python implementation of the "Premium Lux Algo" trading strategy.
    """

    def __init__(self,
                 primary_timeframe: str = "M5",
                 higher_timeframe: str = "H1",
                 supertrend_factor: float = 5.5,
                 smart_trail_atr_period: int = 13,
                 smart_trail_atr_factor: float = 4.0,
                 risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0,
                 **kwargs):
        """
        Initializes the Lux Algo Premium strategy.
        """
        super().__init__(**kwargs)

        self.name = "LuxAlgoPremiumStrategy"
        self.description = "A Python implementation of the 'Premium Lux Algo' TradingView indicator."
        self.version = "1.0.0"

        self.primary_timeframe = primary_timeframe
        self.higher_timeframe = higher_timeframe

        # --- Strategy-Specific Parameters ---
        # These will be set by the profile, but can be passed as arguments
        self.supertrend_factor = supertrend_factor
        self.supertrend_period = 11  # Hardcoded in the original script
        self.confirmation_sma_period = 9  # Hardcoded in the original script
        self.smart_trail_atr_period = smart_trail_atr_period
        self.smart_trail_atr_factor = smart_trail_atr_factor
        self.min_risk_reward = min_risk_reward

        rm_conf = get_risk_manager_config()
        self.risk_percent = rm_conf.get('max_risk_per_trade', risk_percent)

        self._load_timeframe_profile()

        self.processed_bars = {}

    def _load_timeframe_profile(self):
        """Loads parameters from TIMEFRAME_PROFILES based on the primary timeframe."""
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        self.supertrend_factor = profile.get('supertrend_factor', self.supertrend_factor)
        self.smart_trail_atr_period = profile.get('smart_trail_atr_period', self.smart_trail_atr_period)
        self.smart_trail_atr_factor = profile.get('smart_trail_atr_factor', self.smart_trail_atr_factor)

        logger.info(
            f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: "
            f"supertrend_factor={self.supertrend_factor}, "
            f"trail_atr_period={self.smart_trail_atr_period}, trail_atr_factor={self.smart_trail_atr_factor}"
        )

    @property
    def required_timeframes(self) -> List[str]:
        """Specifies the timeframes this strategy needs."""
        return list({self.higher_timeframe, self.primary_timeframe})



    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        **kwargs
    ) -> List[Dict]:
        """The core method where trading signals are generated."""
        if market_data is None:
            market_data = {}
        signals = []
        rm = RiskManager.get_instance()

        for sym, frames in market_data.items():
            higher_df_orig = frames.get(self.higher_timeframe)
            primary_df_orig = frames.get(self.primary_timeframe)

            if not isinstance(higher_df_orig, pd.DataFrame) or higher_df_orig.empty or \
               not isinstance(primary_df_orig, pd.DataFrame) or primary_df_orig.empty:
                logger.debug(f"[{sym}] Missing or empty market data for required timeframes.")
                continue
            
            # Make copies to avoid modifying the original dataframes
            higher_df = higher_df_orig.copy()
            primary_df = primary_df_orig.copy()

            try:
                last_timestamp = str(primary_df.index[-1])
                bar_key = (sym, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_timestamp:
                    continue
                self.processed_bars[bar_key] = last_timestamp
            except IndexError:
                continue
            
            # --- Strategy Logic Pipeline ---

            # 1. Calculate all necessary indicators
            self._calculate_all_indicators(primary_df)
            self._calculate_all_indicators(higher_df, is_higher_tf=True)

            # 2. Determine the market trend using the higher timeframe
            trend = self._determine_trend(higher_df)
            if trend == 'neutral':
                logger.trace(f"[{sym}] HTF trend is neutral, skipping.")
                continue

            # 3. Look for a confirmation signal on the primary timeframe
            confirmation = self._find_confirmation_signal(primary_df)
            if not confirmation:
                continue
            
            direction = confirmation['direction']

            # 4. Confluence Check: Ensure signal aligns with the higher timeframe trend
            if (direction == 'buy' and trend == 'bearish') or \
               (direction == 'sell' and trend == 'bullish'):
                logger.debug(f"[{sym}] Signal '{direction}' misaligned with HTF trend '{trend}'. Skipping.")
                continue

            # 5. Calculate Stop-Loss using the Smart Trail
            stop_loss = self._calculate_smart_trail_sl(primary_df)
            if stop_loss is None:
                logger.warning(f"[{sym}] Could not calculate a valid Smart Trail stop-loss.")
                continue
            
            # 6. Assemble signal and calculate Take-Profit
            entry_price = primary_df['close'].iloc[-1]

            # Final sanity check for SL placement
            if (direction == 'buy' and entry_price <= stop_loss) or \
               (direction == 'sell' and entry_price >= stop_loss):
                logger.debug(f"[{sym}] Entry price {entry_price} is beyond the calculated SL {stop_loss}. Skipping.")
                continue

            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0: continue

            take_profit = entry_price + (risk_per_share * self.min_risk_reward) if direction == 'buy' \
                          else entry_price - (risk_per_share * self.min_risk_reward)
            
            signal_details = {
                "symbol": sym,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": 0.85,
                "description": f"Lux Algo signal based on {trend} trend.",
                "detailed_reasoning": [
                    f"HTF Trend: {trend.capitalize()}",
                    f"Signal: {confirmation['pattern_name']}",
                    f"Smart Trail SL: {stop_loss:.5f}"
                ],
                "signal_timestamp": last_timestamp,
            }

            # 7. Validate the trade with the RiskManager
            validation_result = rm.validate_and_size_trade(signal_details)
            if validation_result['is_valid']:
                logger.info(f"✅ [{sym}] Valid signal generated. Dir: {direction}, Entry: {entry_price:.5f}, SL: {stop_loss:.5f}")
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals

    # ==============================================================================
    # --- Strategy Helper Methods ---
    # ==============================================================================

    def _calculate_all_indicators(self, df: pd.DataFrame, is_higher_tf: bool = False):
        """Calculates and appends all required indicators to the DataFrame."""
        # Main Signal Indicators
        df.ta.supertrend(
            length=self.supertrend_period,
            multiplier=self.supertrend_factor,
            append=True
        )
        df[f'sma_confirm'] = df.ta.sma(length=self.confirmation_sma_period)

        # Smart Trail SL Indicators
        # Use the 'modified' True Range calculation from the script
        hilo = (df['high'] - df['low']).rolling(window=self.smart_trail_atr_period).mean() * 1.5
        href = df['high'] - df['close'].shift(1)
        lref = df['close'].shift(1) - df['low']
        true_range = pd.concat([hilo, href, lref], axis=1).max(axis=1)
        df['wild_ma_tr'] = ta.rma(true_range, length=self.smart_trail_atr_period)

        # HA Trend Filter Indicators (only for higher timeframe)
        if is_higher_tf:
            ha_len = 100  # From script
            o_smooth = ta.ema(df['open'], length=ha_len)
            c_smooth = ta.ema(df['close'], length=ha_len)
            h_smooth = ta.ema(df['high'], length=ha_len)
            l_smooth = ta.ema(df['low'], length=ha_len)

            # Create a temporary DataFrame for Heikin-Ashi calculation
            ha_df = pd.DataFrame({
                'open': o_smooth, 'high': h_smooth, 'low': l_smooth, 'close': c_smooth
            }).dropna()

            if not ha_df.empty:
                # Calculate Heikin-Ashi candles using the correct method name 'ha'.
                # This appends HA_open, HA_high, HA_low, HA_close columns to ha_df.
                ha_df.ta.ha(append=True)
                
                if 'HA_close' in ha_df.columns and 'HA_open' in ha_df.columns:
                    df['ha_osc'] = 100 * (ha_df['HA_close'] - ha_df['HA_open'])
                else:
                    df['ha_osc'] = 0.0
            else:
                df['ha_osc'] = 0.0

    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Analyzes the higher timeframe data to determine the market trend using HA Bias."""
        if 'ha_osc' not in df.columns or df['ha_osc'].isna().all():
            return 'neutral'
        
        last_osc = df['ha_osc'].iloc[-1]
        if last_osc > 0:
            return 'bullish'
        elif last_osc < 0:
            return 'bearish'
        return 'neutral'

    def _find_confirmation_signal(self, df: pd.DataFrame) -> Dict:
        """Looks for the Supertrend crossover/under signal confirmed by SMA."""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        st_col = f"SUPERT_{self.supertrend_period}_{self.supertrend_factor}"
        if st_col not in df.columns or pd.isna(last[st_col]) or pd.isna(last['sma_confirm']):
            return {}

        # Bullish Signal: Crossover supertrend and close is above confirmation SMA
        if prev['close'] <= prev[st_col] and last['close'] > last[st_col] and last['close'] >= last['sma_confirm']:
            return {'direction': 'buy', 'pattern_name': 'Supertrend Crossover Bullish'}

        # Bearish Signal: Crossunder supertrend and close is below confirmation SMA
        if prev['close'] >= prev[st_col] and last['close'] < last[st_col] and last['close'] <= last['sma_confirm']:
            return {'direction': 'sell', 'pattern_name': 'Supertrend Crossunder Bearish'}
            
        return {}

    def _calculate_smart_trail_sl(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculates the Smart Trail stop-loss value for the last bar.
        This function replicates the stateful logic from the Pine Script.
        """
        if 'wild_ma_tr' not in df.columns or df['wild_ma_tr'].isna().all():
            return None

        close = df['close'].to_numpy()
        loss_val = (self.smart_trail_atr_factor * df['wild_ma_tr']).to_numpy()
        
        up_val = close - loss_val
        dn_val = close + loss_val

        n = len(df)
        if n < 2: return None

        trend_up = np.zeros(n)
        trend_down = np.zeros(n)
        trend = np.zeros(n, dtype=int)
        trail = np.zeros(n)

        # Initialization
        trend[0] = 1
        trend_up[0] = up_val[0] if not np.isnan(up_val[0]) else 0
        trend_down[0] = dn_val[0] if not np.isnan(dn_val[0]) else 0
        trail[0] = trend_up[0]

        # Iterative calculation
        for i in range(1, n):
            # Calculate current TrendUp
            if close[i-1] > trend_up[i-1]:
                trend_up[i] = max(up_val[i], trend_up[i-1])
            else:
                trend_up[i] = up_val[i]
            
            # Calculate current TrendDown
            if close[i-1] < trend_down[i-1]:
                trend_down[i] = min(dn_val[i], trend_down[i-1])
            else:
                trend_down[i] = dn_val[i]

            # Determine current Trend
            if close[i] > trend_down[i-1]:
                trend[i] = 1
            elif close[i] < trend_up[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]

            # Determine the trail value based on the trend
            trail[i] = trend_up[i] if trend[i] == 1 else trend_down[i]

        last_trail_value = trail[-1]
        return last_trail_value if np.isfinite(last_trail_value) else None