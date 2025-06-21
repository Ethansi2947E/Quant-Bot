"""
SMC Strategy - V1.4 Implementation with OB Linking

This strategy implements the core logic of Smart Money Concepts, focusing on a
Structure-Based Entry: Trading a pullback to an Order Block after a Break of Structure (BOS).

This version introduces a significant refactoring: The analysis engine now directly links
an Order Block to the Structure Break it originates from. This simplifies the strategy
logic immensely, making it more robust and easier to debug. The detailed checklist
logging remains a core feature.
"""

import pandas as pd
from loguru import logger
from typing import Optional, List, Dict
import numpy as np

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import get_risk_manager_config
from src.risk_manager import RiskManager

# Import our powerful SMC analysis engine
from src.utils.smc_utils import SmartMoneyConcepts, OrderBlock, Structure, Liquidity

# --- Timeframe Mapping ---
TIMEFRAME_MINUTES = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
    'H1': 60, 'H4': 240, 'D1': 1440,
}

# --- Strategy-Specific Parameters ---
SMC_STRATEGY_PARAMS = {
    "risk_percent": 0.01, "min_risk_reward": 3.0,
    "analysis_duration_in_days": 10, "indicator_buffer_period": 200,
    "require_fvg_confluence": False,
    "stop_loss_atr_multiplier": 0.5,
    "atr_period_for_sl": 14
}

class SMCStrategy(SignalGenerator):
    """
    Implements a trading strategy based on Smart Money Concepts with detailed logging.
    """

    def __init__(self, primary_timeframe: str = "M5", **kwargs):
        super().__init__(**kwargs)
        self.name = "SMCStrategy"
        self.description = "A strategy based on Order Blocks, Structure, and Liquidity with direct OB-to-structure linking."
        self.version = "1.4.0"
        self.primary_timeframe = primary_timeframe
        self.params = SMC_STRATEGY_PARAMS.copy()
        self.params.update(kwargs)
        self.lookback_period = self._calculate_adaptive_lookback()
        rm_conf = get_risk_manager_config()
        self.params['risk_percent'] = rm_conf.get('max_risk_per_trade', self.params['risk_percent'])
        self.smc_analyzer = SmartMoneyConcepts()
        self.processed_bars = {}

    @property
    def required_timeframes(self) -> List[str]:
        return [self.primary_timeframe]

    async def initialize(self) -> bool:
        logger.info(f"🔌 Initializing {self.name} v{self.version} on {self.primary_timeframe}")
        return True

    def _calculate_adaptive_lookback(self) -> int:
        duration_days = self.params['analysis_duration_in_days']
        buffer = self.params['indicator_buffer_period']
        minutes_per_bar = TIMEFRAME_MINUTES.get(self.primary_timeframe)
        if not minutes_per_bar:
            logger.error(f"[{self.name}] Unknown timeframe: {self.primary_timeframe}. Defaulting to 1000.")
            return 1000
        analysis_duration_minutes = duration_days * 24 * 60
        required_bars = int(analysis_duration_minutes / minutes_per_bar)
        analysis_window = required_bars + buffer
        final_window = max(250, min(analysis_window, 2500))
        logger.info(
            f"[{self.name}] Adaptive lookback for {self.primary_timeframe}: "
            f"{final_window} bars (for {duration_days} days analysis + {buffer} bar buffer)."
        )
        return final_window

    async def generate_signals(
        self, market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None, **kwargs
    ) -> List[Dict]:
        if market_data is None: market_data = {}
        signals = []
        rm = RiskManager.get_instance()
        import talib
        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)
            min_len = self.smc_analyzer._get_default_config()['swing_length']
            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or len(primary_df) < min_len:
                continue
            if 'atr' not in primary_df.columns:
                primary_df['atr'] = talib.ATR(
                    np.array(primary_df['high']), np.array(primary_df['low']), 
                    np.array(primary_df['close']), 
                    timeperiod=self.smc_analyzer._get_default_config()['atr_period']
                )
            try:
                last_timestamp = str(primary_df.index[-1])
                bar_key = (sym, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_timestamp: continue
                self.processed_bars[bar_key] = last_timestamp
            except IndexError: continue
            
            smc_results = self.smc_analyzer.analyze(primary_df)
            potential_trades = self._find_and_validate_setups(smc_results, sym, primary_df)
            for trade in potential_trades:
                validation_result = rm.validate_and_size_trade(trade)
                if validation_result['is_valid']:
                    logger.success(f"✅ [{sym}] Valid {trade['direction'].upper()} Signal. Reason: {trade['description']}")
                    final_trade_params = validation_result['final_trade_params']
                    signals.append(final_trade_params)
                else:
                    logger.warning(f"❌ Signal for {sym} rejected by RiskManager: {validation_result['reason']}")
        return signals

    def _find_and_validate_setups(self, smc_results: Dict, symbol: str, df: pd.DataFrame) -> List[Dict]:
        valid_setups = []
        last_bar_index = len(df) - 1
        recent_structures = [s for s in smc_results.get('structures', []) if s.break_bar_index > last_bar_index - self.lookback_period]
        for structure in recent_structures:
            if structure.type != 'BOS': continue
            checks = self._get_new_checklist()
            checks['CHECK 1: Break of Structure (BOS) Confirmed'].update({'status': True, 'reason': f"BOS of {structure.pivot.price:.5f} confirmed."})
            
            ob_poi = structure.originating_ob
            if not ob_poi:
                checks['CHECK 2: Order Block (OB) Located']['reason'] = "No originating Order Block was linked to this BOS during analysis."
                self._log_checklist(symbol, "BOS Setup", checks)
                continue
            checks['CHECK 2: Order Block (OB) Located'].update({'status': True, 'reason': f"Found {ob_poi.ob_type} OB @ {ob_poi.low:.5f}-{ob_poi.high:.5f}"})

            has_fvg = self._has_fvg_confluence(ob_poi, smc_results)
            is_required = self.params['require_fvg_confluence']
            
            checks['CHECK 3: Imbalance/FVG Exists']['status'] = has_fvg
            
            if not has_fvg and is_required:
                checks['CHECK 3: Imbalance/FVG Exists']['reason'] = "No FVG confluence found near OB (required)."
                self._log_checklist(symbol, "BOS Setup", checks)
                continue
            elif has_fvg:
                checks['CHECK 3: Imbalance/FVG Exists']['reason'] = "FVG confluence found."
            else:
                checks['CHECK 3: Imbalance/FVG Exists']['reason'] = "FVG confluence not found (optional)."

            has_sweep = self._check_for_liquidity_sweep(ob_poi, smc_results)
            checks['CHECK 4: Liquidity Sweep Precedes OB'].update({'status': has_sweep, 'reason': "OB was preceded by a liquidity sweep." if has_sweep else "No preceding liquidity sweep found."})

            direction = 'buy' if structure.direction == 1 else 'sell'
            entry_price = ob_poi.open
            checks['CHECK 5: Entry Defined'].update({'status': True, 'reason': f"Entry set at OB open: {entry_price:.5f}"})
            
            try:
                atr_at_ob = df['atr'].iloc[ob_poi.bar_index]
                if pd.isna(atr_at_ob): raise ValueError("ATR is NaN")
                sl_buffer = self.params['stop_loss_atr_multiplier'] * atr_at_ob
                stop_loss = (ob_poi.low - sl_buffer) if direction == 'buy' else (ob_poi.high + sl_buffer)
                checks['CHECK 6: Stop Loss Defined'].update({'status': True, 'reason': f"SL set at {stop_loss:.5f} (OB extremity ± {sl_buffer:.5f} ATR buffer)."})
            except (IndexError, TypeError, ValueError) as e:
                checks['CHECK 6: Stop Loss Defined']['reason'] = f"Could not calculate SL due to an error: {e}"
                self._log_checklist(symbol, "BOS Setup", checks)
                continue

            take_profit = structure.pivot.price
            checks['CHECK 7: Take Profit Defined'].update({'status': True, 'reason': f"TP set at broken structure level: {take_profit:.5f}"})

            if stop_loss == entry_price:
                checks['CHECK 8: Risk-to-Reward Ratio Validated']['reason'] = "Entry and SL are the same price."
                self._log_checklist(symbol, "BOS Setup", checks)
                continue
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            if rr_ratio < self.params['min_risk_reward']:
                checks['CHECK 8: Risk-to-Reward Ratio Validated']['reason'] = f"R:R of {rr_ratio:.2f} is below minimum of {self.params['min_risk_reward']:.2f}"
                self._log_checklist(symbol, "BOS Setup", checks)
                continue
            checks['CHECK 8: Risk-to-Reward Ratio Validated'].update({'status': True, 'reason': f"R:R is {rr_ratio:.2f} (min: {self.params['min_risk_reward']:.2f})."})

            self._log_checklist(symbol, "BOS Setup", checks)
            
            critical_checks = ['CHECK 1', 'CHECK 2', 'CHECK 5', 'CHECK 6', 'CHECK 7', 'CHECK 8']
            critical_checks_passed = all(any(k.startswith(cc) and v['status'] for k, v in checks.items()) for cc in critical_checks)
            if self.params['require_fvg_confluence']:
                critical_checks_passed = critical_checks_passed and checks['CHECK 3: Imbalance/FVG Exists']['status']
            
            if critical_checks_passed:
                valid_setups.append({
                    "order_type": "limit", "symbol": symbol, "direction": direction, 
                    "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
                    "timeframe": self.primary_timeframe, "strategy_name": self.name,
                    "confidence": 0.85 if has_fvg and has_sweep else (0.75 if has_fvg or has_sweep else 0.65),
                    "description": f"BOS leading to {ob_poi.ob_type} OB. FVG: {has_fvg}, Sweep: {has_sweep}",
                    "detailed_reasoning": [v['reason'] for k, v in checks.items() if v['status']],
                    "pattern": ob_poi.ob_type, "signal_timestamp": str(ob_poi.time),
                })
        return valid_setups

    def _check_for_liquidity_sweep(self, ob: OrderBlock, smc_results: Dict) -> bool:
        if not hasattr(self.smc_analyzer, 'df') or self.smc_analyzer.df.empty: return False
        if ob.direction == -1:
            for liq_pool in smc_results.get('liquidity_pools', []):
                if liq_pool.direction == -1 and 0 < (ob.bar_index - liq_pool.bar_index) < 20:
                    search_range = slice(liq_pool.bar_index, ob.bar_index + 1)
                    if self.smc_analyzer.df.iloc[search_range]['low'].min() < liq_pool.price: return True
        else:
            for liq_pool in smc_results.get('liquidity_pools', []):
                if liq_pool.direction == 1 and 0 < (ob.bar_index - liq_pool.bar_index) < 20:
                    search_range = slice(liq_pool.bar_index, ob.bar_index + 1)
                    if self.smc_analyzer.df.iloc[search_range]['high'].max() > liq_pool.price: return True
        return False
        
    def _get_new_checklist(self) -> Dict[str, Dict]:
        return {
            "CHECK 1: Break of Structure (BOS) Confirmed": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 2: Order Block (OB) Located": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 3: Imbalance/FVG Exists": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 4: Liquidity Sweep Precedes OB": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 5: Entry Defined": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 6: Stop Loss Defined": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 7: Take Profit Defined": {'status': False, 'reason': 'Not evaluated.'},
            "CHECK 8: Risk-to-Reward Ratio Validated": {'status': False, 'reason': 'Not evaluated.'},
        }

    def _log_checklist(self, symbol: str, setup_type: str, checks: Dict):
        logger.info(f"🔎 [{symbol}] --- {setup_type} Checklist ---")
        for check, result in checks.items():
            status = "✅ PASS" if result['status'] else "❌ FAIL"
            logger.info(f"[{symbol}] {status} | {check}: {result['reason']}")
        logger.info(f"[{symbol}] ---------------------------------")
        
    def _has_fvg_confluence(self, ob: OrderBlock, smc_results: Dict) -> bool:
        for fvg in smc_results.get('fair_value_gaps', []):
            if fvg.direction == -ob.direction and abs(fvg.bar_index - ob.bar_index) < 5:
                return True
        return False
        
    def _check_if_choch_is_sweep(self, choch: Structure, analyzer: "SmartMoneyConcepts") -> bool:
        if choch.direction == 1: return analyzer.swing_low.price < analyzer.prev_swing_low.price
        else: return analyzer.swing_high.price > analyzer.prev_swing_high.price

    def _find_next_liquidity_target(self, direction: int, start_index: int, smc_results: Dict) -> Optional[Liquidity]:
        target_direction = 1 if direction == 1 else -1
        potential_targets = [p for p in smc_results.get('liquidity_pools', []) if p.direction == target_direction and p.bar_index > start_index]
        return min(potential_targets, key=lambda x: x.bar_index) if potential_targets else None

if __name__ == '__main__':
    logger.add("smc_strategy_test.log")
    
    try:
        sample_df = pd.read_csv('test_data_M15.csv', index_col='time', parse_dates=True)
        logger.info(f"Loaded sample data with {len(sample_df)} rows.")
    except FileNotFoundError:
        logger.error("Please create a 'test_data_M15.csv' file for testing.")
        sample_df = pd.DataFrame()

    if not sample_df.empty:
        strategy = SMCStrategy(primary_timeframe="M15")

        market_data_packet = {
            "EUR/USD": {
                "M15": sample_df
            }
        }

        import asyncio
        signals_found = asyncio.run(strategy.generate_signals(market_data=market_data_packet))
        
        if signals_found:
            logger.info(f"🎉 Found {len(signals_found)} signals!")
            for sig in signals_found:
                logger.info(f" -> Signal: {sig['symbol']} {sig['direction']} @ {sig['entry_price']:.5f} | "
                            f"SL: {sig['stop_loss']:.5f} | TP: {sig['take_profit']:.5f}")
                logger.info(f"    Reason: {sig['detailed_reasoning']}")
        else:
            logger.info("No signals generated on the provided data.")