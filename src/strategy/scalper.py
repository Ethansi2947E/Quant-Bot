"""
AlphaQuant Scalper v1 (Python Implementation)

This strategy implements the high-confluence scalping model from Strategy 1.
It is designed for short-horizon trading on high-liquidity crypto assets,
adapting professional concepts like order flow and market structure into
proxies that can be calculated from standard OHLCV data.

- Core Logic: Enters on pullbacks in a confirmed trend.
- Trend: Daily VWAP & EMA crossover.
- Entry Trigger: Price must be in a consolidation phase (RSI 40-60).
- Confirmation: Proxies for buy/sell pressure and interaction with key price levels.

This file adheres to the trading bot's template for seamless integration.
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
import talib

# No longer needed as we are reverting to the proxy method for stability
# from src.features.order_book import get_imbalance


class AlphaQuantScalperV1(SignalGenerator):
    """
    Python implementation of the professional-grade scalping strategy "Strategy 1".
    
    This strategy combines:
    1. Daily VWAP for trend context.
    2. EMA Crossover for momentum.
    3. RSI for identifying non-overextended entry points.
    4. Volume Pressure Proxy (replaces Order Book Imbalance).
    5. Swing Points (replaces Volume Profile Nodes).
    """

    def __init__(self,
                 primary_timeframe: str = "M1",
                 fast_ema_period: int = 9,
                 slow_ema_period: int = 21,
                 rsi_period: int = 14,
                 risk_percent: float = 0.005, # As per strategy spec: 0.5%
                 min_risk_reward: float = 2.0,
                 require_structure_confirmation: bool = False,
                 **kwargs):
        """
        Initializes the AlphaQuant Scalper V1 strategy.

        Args:
            primary_timeframe (str): The main timeframe for signal generation (e.g., "M1", "M5").
            fast_ema_period (int): Period for the fast EMA.
            slow_ema_period (int): Period for the slow EMA.
            rsi_period (int): Period for the RSI.
            risk_percent (float): Percentage of capital to risk per trade.
            min_risk_reward (float): Minimum required risk-to-reward ratio.
            require_structure_confirmation (bool): If True, trades must be near key support/resistance.
        """
        super().__init__(**kwargs)

        self.name = "AlphaQuant Scalper v1 (Python)"
        self.description = "A high-confluence scalping strategy using VWAP, EMAs, RSI, and proxies for order flow and market structure."
        self.version = "1.1.0"

        self.primary_timeframe = primary_timeframe
        
        # --- Strategy-Specific Parameters ---
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.rsi_period = rsi_period
        self.rsi_lower_bound = 40
        self.rsi_upper_bound = 60
        self.min_risk_reward = min_risk_reward
        self.require_structure_confirmation = require_structure_confirmation

        # Reverting to only use the proxy method for stability
        self.imb_proxy_long_thresh = 1.3
        self.imb_proxy_short_thresh = 0.7
        self.imb_proxy_lookback = 10 # Bars to calculate volume pressure over

        # Stop Loss Placement Parameters
        self.swing_lookback_period = 20
        self.time_stop_bars = 5 # This is informational; execution bot handles time stops

        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)
        self.processed_bars = {}

        logger.info(f"✅ [{self.name}] Initialized for {self.primary_timeframe} timeframe.")

    @property
    def required_timeframes(self) -> List[str]:
        """Specifies the timeframes this strategy needs."""
        # This strategy operates purely on the primary timeframe but could be extended.
        return [self.primary_timeframe]

    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        **kwargs
    ) -> List[Dict]:
        """The core method where trading signals are generated based on Strategy 1 rules."""
        if market_data is None:
            return []

        signals = []
        rm = RiskManager.get_instance()

        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)

            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or len(primary_df) < max(self.swing_lookback_period, self.imb_proxy_lookback):
                logger.debug(f"[{sym}] Missing or insufficient primary DataFrame data.")
                continue

            last_timestamp = str(primary_df.index[-1])

            # --- 1. Calculate All Indicators & Proxies ---
            indicators = self._calculate_indicators(primary_df)
            if not indicators:
                continue
            
            logger.trace(f"[{sym}] Calculated Indicators: {indicators}")

            # --- 2. Define Entry Conditions for Long and Short ---
            last_close = indicators['close']
            
            # --- Verbose Condition Checking for Logging ---
            
            # Long conditions
            long_cond_vwap = last_close > indicators['vwap']
            long_cond_ema = indicators['ema_fast'] > indicators['ema_slow']
            long_cond_price_above_ema = last_close > indicators['ema_fast']
            long_cond_rsi = self.rsi_lower_bound <= indicators['rsi'] <= self.rsi_upper_bound
            long_cond_imb = indicators['volume_pressure_ratio'] >= self.imb_proxy_long_thresh
            long_cond_support = indicators['is_near_support']
            
            base_long_conds = [long_cond_vwap, long_cond_ema, long_cond_price_above_ema, long_cond_rsi, long_cond_imb]
            is_long_candidate = all(base_long_conds + ([long_cond_support] if self.require_structure_confirmation else []))

            # Short conditions
            short_cond_vwap = last_close < indicators['vwap']
            short_cond_ema = indicators['ema_fast'] < indicators['ema_slow']
            short_cond_price_below_ema = last_close < indicators['ema_fast']
            short_cond_rsi = self.rsi_lower_bound <= indicators['rsi'] <= self.rsi_upper_bound
            short_cond_imb = indicators['volume_pressure_ratio'] <= self.imb_proxy_short_thresh
            short_cond_resistance = indicators['is_near_resistance']

            base_short_conds = [short_cond_vwap, short_cond_ema, short_cond_price_below_ema, short_cond_rsi, short_cond_imb]
            is_short_candidate = all(base_short_conds + ([short_cond_resistance] if self.require_structure_confirmation else []))

            # --- Structured Logging ---
            long_structure_icon = '✅' if long_cond_support else ('❌' if self.require_structure_confirmation else '⚠️')
            short_structure_icon = '✅' if short_cond_resistance else ('❌' if self.require_structure_confirmation else '⚠️')

            log_msg = f"[{sym}] Condition Analysis for Bar {last_timestamp}:\n"
            log_msg += f"  - LONG -> {'CANDIDATE' if is_long_candidate else 'REJECTED'}\n"
            log_msg += f"    {'✅' if long_cond_vwap else '❌'} [Trend]     Price > VWAP: {last_close:.2f} > {indicators['vwap']:.2f}\n"
            log_msg += f"    {'✅' if long_cond_ema else '❌'} [Momentum]  EMA Fast > Slow\n"
            log_msg += f"    {'✅' if long_cond_price_above_ema else '❌'} [Filter]    Price > Fast EMA\n"
            log_msg += f"    {'✅' if long_cond_rsi else '❌'} [Entry]     RSI in range: {self.rsi_lower_bound} <= {indicators['rsi']:.2f} <= {self.rsi_upper_bound}\n"
            log_msg += f"    {'✅' if long_cond_imb else '❌'} [Pressure]  Buy Ratio >= {self.imb_proxy_long_thresh}: {indicators['volume_pressure_ratio']:.2f}\n"
            log_msg += f"    {long_structure_icon} [Structure] Price near Support\n"
            
            log_msg += f"  - SHORT -> {'CANDIDATE' if is_short_candidate else 'REJECTED'}\n"
            log_msg += f"    {'✅' if short_cond_vwap else '❌'} [Trend]     Price < VWAP: {last_close:.2f} < {indicators['vwap']:.2f}\n"
            log_msg += f"    {'✅' if short_cond_ema else '❌'} [Momentum]  EMA Fast < Slow\n"
            log_msg += f"    {'✅' if short_cond_price_below_ema else '❌'} [Filter]    Price < Fast EMA\n"
            log_msg += f"    {'✅' if short_cond_rsi else '❌'} [Entry]     RSI in range: {self.rsi_lower_bound} <= {indicators['rsi']:.2f} <= {self.rsi_upper_bound}\n"
            log_msg += f"    {'✅' if short_cond_imb else '❌'} [Pressure]  Sell Ratio <= {self.imb_proxy_short_thresh}: {indicators['volume_pressure_ratio']:.2f}\n"
            log_msg += f"    {short_structure_icon} [Structure] Price near Resistance"

            logger.info(log_msg)
            
            # --- 3. Assemble and Validate Signal ---
            if is_long_candidate:
                direction = 'buy'
                entry = last_close
                stop_loss = indicators['last_swing_low']
                # Ensure stop loss is not too close
                if entry - stop_loss < indicators['atr'] * 0.5:
                    stop_loss = entry - indicators['atr'] * 0.5
                take_profit = entry + (entry - stop_loss) * self.min_risk_reward
                
                reasoning = [
                    f"Trend Confirmed: Price > VWAP ({last_close:.2f} > {indicators['vwap']:.2f})",
                    f"Momentum Bullish: EMA{self.fast_ema_period} > EMA{self.slow_ema_period}",
                    f"Pullback Entry: RSI at {indicators['rsi']:.2f} (in 40-60 range)",
                    f"Buy Pressure Proxy: Volume Ratio is {indicators['volume_pressure_ratio']:.2f} (>= {self.imb_proxy_long_thresh})",
                ]
                if long_cond_support:
                    reasoning.append(f"Structural Support: Price near swing low at {indicators['last_swing_low']:.2f}")
                
            elif is_short_candidate:
                direction = 'sell'
                entry = last_close
                stop_loss = indicators['last_swing_high']
                # Ensure stop loss is not too close
                if stop_loss - entry < indicators['atr'] * 0.5:
                    stop_loss = entry + indicators['atr'] * 0.5
                take_profit = entry - (stop_loss - entry) * self.min_risk_reward

                reasoning = [
                    f"Trend Confirmed: Price < VWAP ({last_close:.2f} < {indicators['vwap']:.2f})",
                    f"Momentum Bearish: EMA{self.fast_ema_period} < EMA{self.slow_ema_period}",
                    f"Rally Entry: RSI at {indicators['rsi']:.2f} (in 40-60 range)",
                    f"Sell Pressure Proxy: Volume Ratio is {indicators['volume_pressure_ratio']:.2f} (<= {self.imb_proxy_short_thresh})",
                ]
                if short_cond_resistance:
                    reasoning.append(f"Structural Resistance: Price near swing high at {indicators['last_swing_high']:.2f}")

            else:
                continue # No valid signal

            # --- 4. Package and Validate with Risk Manager ---
            logger.info(f"[{sym}] Found valid {direction.upper()} candidate. Packaging signal for risk validation.")
            signal_details = {
                "symbol": sym,
                "direction": direction,
                "entry_price": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": 0.85, # High confidence due to multi-factor confluence
                "detailed_reasoning": reasoning,
                "signal_timestamp": last_timestamp,
            }
            
            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.success(f"✅ [{sym}] Valid {direction.upper()} signal. Entry: {entry:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"❌ Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals

    # ==============================================================================
    # --- HELPER METHODS for calculating indicators and proxies ---
    # ==============================================================================

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculates all necessary indicators and returns them in a dictionary."""
        # Work on a copy to avoid SettingWithCopyWarning and other side effects.
        data = df.copy()
        try:
            # Standard Indicators
            data['ema_fast'] = talib.EMA(data['close'].to_numpy(dtype=float), timeperiod=self.fast_ema_period)
            data['ema_slow'] = talib.EMA(data['close'].to_numpy(dtype=float), timeperiod=self.slow_ema_period)
            data['rsi'] = talib.RSI(data['close'].to_numpy(dtype=float), timeperiod=self.rsi_period)
            data['atr'] = talib.ATR(data['high'].to_numpy(dtype=float), data['low'].to_numpy(dtype=float), data['close'].to_numpy(dtype=float), timeperiod=14)
            
            # Custom VWAP Calculation
            data = self._get_daily_vwap(data)
            
            # Custom Order Book Imbalance Proxy
            data['volume_pressure_ratio'] = self._get_volume_pressure_proxy(data, self.imb_proxy_lookback)

            # Key Levels (Proxy for Volume Profile Nodes)
            last_swing_high = data['high'].iloc[-self.swing_lookback_period:-1].max()
            last_swing_low = data['low'].iloc[-self.swing_lookback_period:-1].min()
            
            last_candle = data.iloc[-1]
            atr = last_candle['atr']
            
            return {
                "close": last_candle['close'],
                "vwap": last_candle['vwap'],
                "ema_fast": last_candle['ema_fast'],
                "ema_slow": last_candle['ema_slow'],
                "rsi": last_candle['rsi'],
                "atr": atr,
                "volume_pressure_ratio": last_candle['volume_pressure_ratio'],
                "last_swing_high": last_swing_high,
                "last_swing_low": last_swing_low,
                "is_near_support": abs(last_candle['close'] - last_swing_low) <= atr * 0.5,
                "is_near_resistance": abs(last_candle['close'] - last_swing_high) <= atr * 0.5
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def _get_daily_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates and appends the daily VWAP to the DataFrame."""
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['typical_price'] * df['volume']
        
        # Group by date and calculate cumulative sums for VWAP
        # Using pd.to_datetime makes the index type explicit for the linter.
        grouped = df.groupby(pd.to_datetime(df.index).date)
        cum_vol = grouped['volume'].cumsum()
        cum_tp_vol = grouped['tp_vol'].cumsum()
        
        df['vwap'] = cum_tp_vol / cum_vol
        return df.drop(columns=['typical_price', 'tp_vol'])

    def _get_volume_pressure_proxy(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """
        Proxy for Order Book Imbalance.
        Calculates the ratio of buying volume to selling volume over a lookback period.
        - Buying Volume: Volume on candles that closed higher than they opened.
        - Selling Volume: Volume on candles that closed lower than they opened.
        A ratio > 1 indicates more buying pressure; < 1 indicates more selling pressure.
        """
        buy_vol = df['volume'].where(df['close'] > df['open'], 0)
        sell_vol = df['volume'].where(df['close'] < df['open'], 0)

        sum_buy_vol = buy_vol.rolling(window=lookback, min_periods=1).sum()
        sum_sell_vol = sell_vol.rolling(window=lookback, min_periods=1).sum()

        # Add a small epsilon to avoid division by zero
        return sum_buy_vol / (sum_sell_vol + 1e-10)
