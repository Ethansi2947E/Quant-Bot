'''
Dante Bills Strategy

This is a Python implementation of the 'Dante_Bills' Pine Script strategy.
It is designed to be a 1-to-1 conversion, maintaining the original logic and parameters.

The core logic is a crossover of two moving averages, one based on the 'close' price
and the other on the 'open' price. The strategy can optionally use a higher timeframe
for the signal, similar to Pine Script's `request.security` function.
'''

import pandas as pd
import pandas_ta as ta
from loguru import logger
from typing import Optional, List, Dict, Any, Tuple

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import RISK_MANAGER_CONFIG
from src.risk_manager import RiskManager

class DanteBillsStrategy(SignalGenerator):
    """
    A Python conversion of the 'Dante_Bills' Pine Script strategy.
    """

    def __init__(self,
                 primary_timeframe: str = "15",
                 use_alternate_resolution: bool = True,
                 alternate_resolution_multiplier: int = 8,
                 basis_type: str = 'ALMA',
                 basis_len: int = 2,
                 offset_sigma: int = 5,
                 offset_alma: float = 0.85,
                 delay_offset: int = 0,
                 trade_type: str = 'BOTH',
                 stop_loss_points: int = 0,
                 take_profit_points: int = 0,
                 risk_percent: float = 0.10, # Default from strategy is 10% of equity
                 **kwargs):
        """
        Initializes the DanteBillsStrategy with parameters from the original Pine Script.
        """
        super().__init__(**kwargs)

        # --- Strategy Identity ---
        self.name = "DanteBillsStrategy"
        self.description = "A Python conversion of the Dante_Bills Pine Script strategy."
        self.version = "1.0.0"

        # --- Timeframe Parameters ---
        self.primary_timeframe = primary_timeframe
        self.use_alternate_resolution = use_alternate_resolution
        
        # Logic to construct the higher timeframe string, simplified from Pine Script
        # Note: This requires the user to provide a valid timeframe string like '1H', '4H', etc.
        # A full conversion of the Pine Script logic is complex and environment-dependent.
        self.higher_timeframe = f"{int(primary_timeframe) * alternate_resolution_multiplier}T" if 'T' in primary_timeframe.upper() else f"{int(primary_timeframe) * alternate_resolution_multiplier}H"

        # --- Indicator & Strategy Parameters ---
        self.basis_type = basis_type
        self.basis_len = basis_len
        self.offset_sigma = offset_sigma
        self.offset_alma = offset_alma
        self.delay_offset = delay_offset
        self.trade_type = trade_type
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points

        # --- Risk Management ---
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)
        self.lookback = self.basis_len + 5 # Basic lookback for MA calculations

    @property
    def lookback_periods(self) -> Dict[str, int]:
        """Returns the lookback period required for the strategy."""
        return {self.primary_timeframe: self.lookback}

    @property
    def required_timeframes(self) -> List[str]:
        """Specifies the timeframes needed for this strategy."""
        if self.use_alternate_resolution:
            return list({self.primary_timeframe, self.higher_timeframe})
        return [self.primary_timeframe]

    def _variant_ma(self, source: pd.Series) -> pd.Series:
        """
        Python implementation of the 'variant' function from Pine Script.
        Calculates one of many moving average types based on the strategy's settings.
        Uses the pandas_ta library for most calculations.
        """
        ma_type = self.basis_type.upper()
        length = self.basis_len

        if ma_type == 'SMA':
            return ta.sma(source, length=length)
        elif ma_type == 'EMA':
            return ta.ema(source, length=length)
        elif ma_type == 'DEMA':
            return ta.dema(source, length=length)
        elif ma_type == 'TEMA':
            return ta.tema(source, length=length)
        elif ma_type == 'WMA':
            return ta.wma(source, length=length)
        elif ma_type == 'VWMA':
            return ta.vwma(source, self.data['volume'], length=length)
        elif ma_type == 'HULLMA':
            return ta.hma(source, length=length)
        elif ma_type == 'LSMA':
            return ta.linreg(source, length=length, offset=self.offset_sigma)
        elif ma_type == 'ALMA':
            return ta.alma(source, length=length, sigma=self.offset_sigma, offset=self.offset_alma)
        # Add other MAs as needed...
        else:
            logger.warning(f"Unsupported MA type '{self.basis_type}'. Defaulting to SMA.")
            return ta.sma(source, length=length)

    def prepare_data(self, market_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """(Backtesting Only) Pre-calculates all indicators for the entire dataset."""
        logger.info(f"[{self.name}] Preparing data for backtesting...")
        prepared_data = market_data.copy()

        for sym, frames in prepared_data.items():
            primary_df = frames.get(self.primary_timeframe)
            if primary_df is None:
                continue

            # Apply delay offset from inputs
            close_delayed = primary_df['close'].shift(self.delay_offset)
            open_delayed = primary_df['open'].shift(self.delay_offset)

            # Calculate MAs on the primary timeframe
            close_series = self._variant_ma(close_delayed)
            open_series = self._variant_ma(open_delayed)

            if self.use_alternate_resolution:
                higher_df = frames.get(self.higher_timeframe)
                if higher_df is None:
                    logger.warning(f"Higher timeframe '{self.higher_timeframe}' not found for {sym}. Skipping alternate resolution.")
                    primary_df['closeSeriesAlt'] = close_series
                    primary_df['openSeriesAlt'] = open_series
                else:
                    # Calculate MAs on higher timeframe
                    htf_close_delayed = higher_df['close'].shift(self.delay_offset)
                    htf_open_delayed = higher_df['open'].shift(self.delay_offset)
                    htf_close_series = self._variant_ma(htf_close_delayed)
                    htf_open_series = self._variant_ma(htf_open_delayed)

                    # Map higher timeframe indicators to primary timeframe index (like request.security)
                    primary_df['closeSeriesAlt'] = htf_close_series.reindex(primary_df.index, method='ffill')
                    primary_df['openSeriesAlt'] = htf_open_series.reindex(primary_df.index, method='ffill')
            else:
                primary_df['closeSeriesAlt'] = close_series
                primary_df['openSeriesAlt'] = open_series

            # Calculate crossover/crossunder signals
            close_alt = primary_df['closeSeriesAlt']
            open_alt = primary_df['openSeriesAlt']
            primary_df['buy_signal'] = (close_alt > open_alt) & (close_alt.shift(1) <= open_alt.shift(1))
            primary_df['sell_signal'] = (close_alt < open_alt) & (close_alt.shift(1) >= open_alt.shift(1))

            primary_df.dropna(inplace=True)
            frames[self.primary_timeframe] = primary_df

        return prepared_data

    async def generate_signals(self, market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None, **kwargs) -> List[Dict]:
        if market_data is None: market_data = {}
        signals = []
        rm = RiskManager.get_instance()

        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)
            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or len(primary_df) < self.lookback:
                continue

            # Check if indicators are pre-calculated (backtesting) or need to be calculated (live)
            indicator_cols = ['buy_signal', 'sell_signal']
            if not all(col in primary_df.columns for col in indicator_cols):
                # This block is for live trading. It recalculates on the small, recent data slice.
                live_data = self.prepare_data({sym: {self.primary_timeframe: primary_df}})
                primary_df = live_data[sym][self.primary_timeframe]

            if primary_df.empty:
                continue

            last = primary_df.iloc[-1]
            direction = None

            if last['buy_signal'] and self.trade_type in ['BOTH', 'LONG']:
                direction = 'buy'
            elif last['sell_signal'] and self.trade_type in ['BOTH', 'SHORT']:
                direction = 'sell'

            if not direction:
                continue

            # --- Assemble Signal ---
            entry = last['close']
            
            # Simplified point calculation for SL/TP. Assumes a standard forex pair.
            # A robust solution would get this from symbol properties.
            point_value = 0.00001 

            if direction == 'buy':
                stop_loss = entry - (self.stop_loss_points * point_value) if self.stop_loss_points > 0 else entry * 0.99 # Fallback SL
                take_profit = entry + (self.take_profit_points * point_value) if self.take_profit_points > 0 else entry * 1.02 # Fallback TP
            else: # sell
                stop_loss = entry + (self.stop_loss_points * point_value) if self.stop_loss_points > 0 else entry * 1.01 # Fallback SL
                take_profit = entry - (self.take_profit_points * point_value) if self.take_profit_points > 0 else entry * 0.98 # Fallback TP

            signal_details = {
                "symbol": sym,
                "direction": direction,
                "entry_price": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": 0.80, # Base confidence for a valid crossover
                "detailed_reasoning": [f"MA Crossover signal on {self.primary_timeframe}"],
                "signal_timestamp": str(last.name),
            }

            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.info(f"✅ [{sym}] Valid {self.name} signal. Dir: {direction}, Entry: {entry:.5f}")
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"Signal for {sym} from {self.name} rejected by RiskManager: {validation_result['reason']}")

        return signals
