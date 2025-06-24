"""
Lux Algo Premium Strategy (1:1 Mirror)

This strategy is a direct Python implementation of the core logic from the "Premium Lux Algo"
TradingView indicator. It now perfectly mirrors the original script's behavior by
performing all calculations (entry signal, trend filter, and exit logic) on a
SINGLE timeframe.

Core Logic:
- Entry Signal: A crossover/crossunder of the price and a Supertrend line, confirmed
  by a simple moving average.
- Trend Filter: Uses a smoothed Heikin-Ashi indicator on the SAME timeframe to
  ensure signals align with the immediate trend.
- Exit Logic: Implements the "Smart Trail", a sophisticated trailing stop-loss based
  on a Wilder's Moving Average of the True Range. Take-profit is calculated based
  on a fixed risk-to-reward ratio.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, List, Dict

# Use pandas-ta for a wider range of indicators that map well to TradingView
try:
    import pandas_ta as ta
except ImportError:
    logger.error("The 'pandas-ta' library is required for this strategy. Please install it using 'pip install pandas-ta'")
    exit()

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import get_risk_manager_config
from src.risk_manager import RiskManager

# --- Timeframe-Specific Parameter Profiles ---
# These profiles adjust the strategy's sensitivity and responsiveness for different timeframes.
TIMEFRAME_PROFILES = {
    "M1": {"supertrend_factor": 7.0, "smart_trail_atr_period": 15, "smart_trail_atr_factor": 5.0}, # Added M1
    "M5": {"supertrend_factor": 6.0, "smart_trail_atr_period": 14, "smart_trail_atr_factor": 4.5},
    "M15": {"supertrend_factor": 5.5, "smart_trail_atr_period": 13, "smart_trail_atr_factor": 4.0},
    "H1": {"supertrend_factor": 5.0, "smart_trail_atr_period": 12, "smart_trail_atr_factor": 3.5},
    "H4": {"supertrend_factor": 4.5, "smart_trail_atr_period": 11, "smart_trail_atr_factor": 3.0},
}
DEFAULT_PROFILE = TIMEFRAME_PROFILES["M15"]


class LuxAlgoPremiumStrategy2(SignalGenerator):
    """
    A 1:1 Python implementation of the "Premium Lux Algo" trading strategy.
    Operates on a single timeframe.
    """

    def __init__(self,
                 primary_timeframe: str = "M5",
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

        self.name = "LuxAlgoPremiumStrategy2"
        self.description = "A 1:1 Python implementation of the 'Premium Lux Algo' TradingView indicator on a single timeframe."
        self.version = "1.2.0" # Version updated for M1 parameters

        self.primary_timeframe = primary_timeframe

        # --- Strategy-Specific Parameters ---
        self.supertrend_factor = supertrend_factor
        self.supertrend_period = 11
        self.confirmation_sma_period = 9
        self.smart_trail_atr_period = smart_trail_atr_period
        self.smart_trail_atr_factor = smart_trail_atr_factor
        self.min_risk_reward = min_risk_reward

        rm_conf = get_risk_manager_config()
        self.risk_percent = rm_conf.get('max_risk_per_trade', risk_percent)

        self._load_timeframe_profile()

        self.processed_bars = {}

    def _load_timeframe_profile(self):
        """Loads parameters from TIMEFRAME_PROFILES based on the strategy's timeframe."""
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
        """Specifies the single timeframe this strategy needs."""
        return [self.primary_timeframe]

    async def initialize(self) -> bool:
        """Optional setup method."""
        logger.info(f"🔌 Initializing {self.name}")
        return True

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
            df_orig = frames.get(self.primary_timeframe)

            if not isinstance(df_orig, pd.DataFrame) or df_orig.empty:
                logger.debug(f"[{sym}] Missing or empty market data for timeframe {self.primary_timeframe}.")
                continue
            
            df = df_orig.copy()

            try:
                last_timestamp = str(df.index[-1])
                bar_key = (sym, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_timestamp:
                    continue
                self.processed_bars[bar_key] = last_timestamp
            except IndexError:
                continue
            
            # --- Strategy Logic Pipeline ---

            # 1. Calculate all necessary indicators on the single timeframe
            self._calculate_all_indicators(df)

            # 2. Determine the market trend using the same timeframe
            trend = self._determine_trend(df)
            if trend == 'neutral':
                logger.trace(f"[{sym}] Trend on {self.primary_timeframe} is neutral, skipping.")
                continue

            # 3. Look for a confirmation signal
            confirmation = self._find_confirmation_signal(df)
            if not confirmation:
                continue
            
            direction = confirmation['direction']

            # 4. Confluence Check: Ensure signal aligns with the timeframe's trend
            if (direction == 'buy' and trend == 'bearish') or \
               (direction == 'sell' and trend == 'bullish'):
                logger.debug(f"[{sym}] Signal '{direction}' misaligned with {self.primary_timeframe} trend '{trend}'. Skipping.")
                continue

            # 5. Calculate Stop-Loss using the Smart Trail
            stop_loss = self._calculate_smart_trail_sl(df)
            if stop_loss is None:
                logger.warning(f"[{sym}] Could not calculate a valid Smart Trail stop-loss.")
                continue
            
            # 6. Assemble signal and calculate Take-Profit
            entry_price = df['close'].iloc[-1]

            if (direction == 'buy' and entry_price <= stop_loss) or \
               (direction == 'sell' and entry_price >= stop_loss):
                logger.debug(f"[{sym}] Entry price {entry_price} is beyond the calculated SL {stop_loss}. Skipping.")
                continue

            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0: continue

            take_profit = entry_price + (risk_per_share * self.min_risk_reward) if direction == 'buy' \
                          else entry_price - (risk_per_share * self.min_risk_reward)
            
            signal_details = {
                "symbol": sym, "direction": direction, "entry_price": entry_price,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "timeframe": self.primary_timeframe, "strategy_name": self.name,
                "confidence": 0.80,
                "description": f"Lux Algo signal based on {self.primary_timeframe} trend.",
                "detailed_reasoning": [
                    f"Timeframe Trend: {trend.capitalize()}",
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

    def _calculate_all_indicators(self, df: pd.DataFrame):
        """Calculates and appends all required indicators to the DataFrame."""
        # Main Signal Indicators
        df.ta.supertrend(
            length=self.supertrend_period,
            multiplier=self.supertrend_factor,
            append=True
        )
        df[f'sma_confirm'] = df.ta.sma(length=self.confirmation_sma_period)

        # Smart Trail SL Indicators
        hilo = (df['high'] - df['low']).rolling(window=self.smart_trail_atr_period).mean() * 1.5
        href = df['high'] - df['close'].shift(1)
        lref = df['close'].shift(1) - df['low']
        true_range = pd.concat([hilo, href, lref], axis=1).max(axis=1)
        df['wild_ma_tr'] = ta.rma(true_range, length=self.smart_trail_atr_period)

        # HA Trend Filter Indicators (now calculated on the same timeframe)
        ha_len = 100
        o_smooth = ta.ema(df['open'], length=ha_len)
        c_smooth = ta.ema(df['close'], length=ha_len)
        h_smooth = ta.ema(df['high'], length=ha_len)
        l_smooth = ta.ema(df['low'], length=ha_len)
        
        ha_df = pd.DataFrame({
            'open': o_smooth, 'high': h_smooth, 'low': l_smooth, 'close': c_smooth
        }).dropna()
        
        if not ha_df.empty:
            # Correctly calculate Heikin-Ashi using the 'ha' method
            ha_df.ta.ha(append=True)
            if 'HA_close' in ha_df.columns and 'HA_open' in ha_df.columns:
                # Use reindex to safely align the oscillator with the original DataFrame
                ha_osc = 100 * (ha_df['HA_close'] - ha_df['HA_open'])
                df['ha_osc'] = ha_osc.reindex(df.index, method='ffill')
            else:
                df['ha_osc'] = 0.0
        else:
            df['ha_osc'] = 0.0

    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Analyzes the DataFrame to determine the market trend using HA Bias."""
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
        if len(df) < 2: return {}
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        st_col = f"SUPERT_{self.supertrend_period}_{self.supertrend_factor}"
        if st_col not in df.columns or pd.isna(last[st_col]) or pd.isna(last['sma_confirm']):
            return {}

        if prev['close'] <= prev[st_col] and last['close'] > last[st_col] and last['close'] >= last['sma_confirm']:
            return {'direction': 'buy', 'pattern_name': 'Supertrend Crossover Bullish'}

        if prev['close'] >= prev[st_col] and last['close'] < last[st_col] and last['close'] <= last['sma_confirm']:
            return {'direction': 'sell', 'pattern_name': 'Supertrend Crossunder Bearish'}
            
        return {}

    def _calculate_smart_trail_sl(self, df: pd.DataFrame) -> Optional[float]:
        """Calculates the Smart Trail stop-loss value for the last bar."""
        if 'wild_ma_tr' not in df.columns or df['wild_ma_tr'].isna().all():
            return None

        close = df['close'].to_numpy()
        loss_val = (self.smart_trail_atr_factor * df['wild_ma_tr']).to_numpy()
        
        up_val = close - loss_val
        dn_val = close + loss_val

        n = len(df)
        if n < 2: return None

        trend_up, trend_down, trail = np.zeros(n), np.zeros(n), np.zeros(n)
        trend = np.zeros(n, dtype=int)

        trend[0] = 1
        trend_up[0] = up_val[0] if not np.isnan(up_val[0]) else 0
        trend_down[0] = dn_val[0] if not np.isnan(dn_val[0]) else 0
        trail[0] = trend_up[0]

        for i in range(1, n):
            if close[i-1] > trend_up[i-1]:
                trend_up[i] = max(up_val[i], trend_up[i-1])
            else:
                trend_up[i] = up_val[i]
            
            if close[i-1] < trend_down[i-1]:
                trend_down[i] = min(dn_val[i], trend_down[i-1])
            else:
                trend_down[i] = dn_val[i]

            if close[i] > trend_down[i-1]:
                trend[i] = 1
            elif close[i] < trend_up[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]

            trail[i] = trend_up[i] if trend[i] == 1 else trend_down[i]

        last_trail_value = trail[-1]
        return last_trail_value if np.isfinite(last_trail_value) else None