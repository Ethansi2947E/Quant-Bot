"""
SMC Strategy - V1.2 Implementation

This strategy implements the core logic of Smart Money Concepts, focusing on two primary entry models:
1.  Structure-Based Entry: Trading a pullback to an Order Block after a Break of Structure (BOS).
2.  Liquidity-Based Entry: Trading a reversal from an Order Block created after a liquidity sweep and a subsequent Change of Character (CHoCH).

It uses the `SmartMoneyConcepts` utility with its default configuration to perform the underlying market structure analysis.
"""

import pandas as pd
from loguru import logger
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import get_risk_manager_config
from src.risk_manager import RiskManager

# Import our powerful SMC analysis engine
from src.utils.smc_utils import SmartMoneyConcepts, OrderBlock, Structure, Liquidity, FairValueGap

# --- Best Practice: Define Strategy-Specific Parameters ---
# These parameters control the STRATEGY'S behavior, not the underlying SMC analyzer's.
SMC_STRATEGY_PARAMS = {
    # --- Strategy Risk & Trade Management ---
    "risk_percent": 0.01,  # 1% risk per trade
    "min_risk_reward": 3.0, # Minimum R:R ratio to consider a trade
    
    # --- Logic Controls ---
    "lookback_period": 1000, # How many bars back to look for recent signals
    "require_fvg_confluence": True, # If True, an OB must have a related FVG to be considered
    "stop_loss_atr_multiplier": 0.5, # SL = OB low/high +/- (X * ATR)
    "atr_period_for_sl": 14 # A standard ATR period for stop loss calculation
}

class SMCStrategy(SignalGenerator):
    """
    Implements a trading strategy based on Smart Money Concepts.
    """

    def __init__(self,
                 primary_timeframe: str = "M15",
                 **kwargs):
        """
        Initializes the SMC strategy.

        Args:
            primary_timeframe (str): The main timeframe for signal generation and execution.
            **kwargs: Catches any additional keyword arguments for base class.
        """
        # --- 1. Call the parent constructor ---
        super().__init__(**kwargs)

        # --- 2. Set Strategy Identity ---
        self.name = "SMCStrategy"
        self.description = "A strategy based on Order Blocks, Structure, and Liquidity."
        self.version = "1.2.0" # Version incremented to reflect refactoring

        # --- 3. Set Timeframe Parameters ---
        self.primary_timeframe = primary_timeframe

        # --- 4. Load Strategy Parameters ---
        self.params = SMC_STRATEGY_PARAMS.copy()
        # Allow overriding strategy params from kwargs if needed
        self.params.update(kwargs)
        
        # --- EXPOSE a copy of the lookback_period for the bot to use ---
        self.lookback_period = self.params['lookback_period']

        rm_conf = get_risk_manager_config()
        self.params['risk_percent'] = rm_conf.get('max_risk_per_trade', self.params['risk_percent'])

        # --- 5. Initialize the SMC Analyzer ---
        # CLEANUP: We now instantiate the analyzer with NO config argument.
        # This forces it to use its own internal `_get_default_config()` method.
        # The strategy is now decoupled from the analyzer's internal settings.
        self.smc_analyzer = SmartMoneyConcepts()

        # --- 6. Initialize State Tracking ---
        self.processed_bars = {}

    @property
    def required_timeframes(self) -> List[str]:
        """Specifies that this strategy only needs its primary timeframe."""
        return [self.primary_timeframe]

    async def initialize(self) -> bool:
        logger.info(f"🔌 Initializing {self.name} on {self.primary_timeframe}")
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
        
        # We need to calculate ATR for our SL, but the SMC analyzer does it internally.
        # Instead of running it twice, let's just ensure the primary_df has it.
        # This is a good example of ensuring data prerequisites.
        import talib

        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)
            
            # Use the default swing_length from the analyzer to check df length
            min_len = self.smc_analyzer._get_default_config()['swing_length']
            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or len(primary_df) < min_len:
                continue

            # --- Data Integrity Check ---
            # Ensure the ATR column exists for our SL calculation.
            # The smc_analyzer will also create an 'atr' column if it doesn't exist.
            if 'atr' not in primary_df.columns:
                 primary_df['atr'] = talib.ATR(np.array(primary_df['high']), np.array(primary_df['low']), np.array(primary_df['close']), 
                                              timeperiod=self.smc_analyzer._get_default_config()['atr_period'])

            try:
                last_timestamp = str(primary_df.index[-1])
                bar_key = (sym, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_timestamp:
                    continue
                self.processed_bars[bar_key] = last_timestamp
            except IndexError:
                continue

            # --- 1. Perform SMC Analysis ---
            smc_results = self.smc_analyzer.analyze(primary_df)

            # --- 2. Find Potential Trade Setups ---
            potential_trades = self._find_potential_setups(smc_results, sym, primary_df)

            if not potential_trades:
                continue

            # --- 3. Assemble and Validate Signals ---
            for trade in potential_trades:
                signal_details = self._assemble_signal(trade, sym, primary_df)
                if not signal_details:
                    continue

                validation_result = rm.validate_and_size_trade(signal_details)

                if validation_result['is_valid']:
                    logger.success(f"✅ [{sym}] Valid {trade['type']} Signal. Reason: {trade['reasoning'][0]}")
                    final_trade_params = validation_result['final_trade_params']
                    signals.append(final_trade_params)
                else:
                    logger.warning(f"❌ Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals

    # ==============================================================================
    # --- Core Strategy Logic Helper Methods ---
    # ==============================================================================

    def _find_potential_setups(self, smc_results: Dict, symbol: str, df: pd.DataFrame) -> List[Dict]:
        setups = []
        last_bar_index = len(df) - 1

        recent_structures = [s for s in smc_results.get('structures', []) 
                             if s.break_bar_index > last_bar_index - self.params['lookback_period']]
        
        for structure in recent_structures:
            if structure.type == 'BOS':
                ob_candidates = smc_results.get('swing_order_blocks', []) + smc_results.get('internal_order_blocks', [])
                for ob in sorted(ob_candidates, key=lambda x: x.bar_index, reverse=True):
                    if (ob.bar_index < structure.break_bar_index and 
                        ob.direction == -structure.direction):
                        if self.params['require_fvg_confluence'] and not self._has_fvg_confluence(ob, smc_results):
                            continue
                        setups.append({
                            "poi": ob,
                            "direction": 'buy' if structure.direction == 1 else 'sell',
                            "target_liquidity": structure.pivot.price,
                            "type": "Structure-Based (BOS)",
                            "reasoning": [f"BOS at {structure.pivot.price:.5f}", f"POI: {ob.ob_type} OB at {ob.low:.5f}-{ob.high:.5f}"]
                        })
                        break 
        
        for structure in recent_structures:
            if structure.type == 'CHoCH':
                is_sweep = self._check_if_choch_is_sweep(structure, self.smc_analyzer)
                if not is_sweep:
                    continue

                ob_candidates = smc_results.get('swing_order_blocks', []) + smc_results.get('internal_order_blocks', [])
                for ob in sorted(ob_candidates, key=lambda x: x.bar_index, reverse=True):
                    if ob.ob_type == 'Extreme' and ob.bar_index < structure.break_bar_index and ob.direction == -structure.direction:
                        if self.params['require_fvg_confluence'] and not self._has_fvg_confluence(ob, smc_results):
                            continue
                        
                        next_liq_pool = self._find_next_liquidity_target(structure.direction, ob.bar_index, smc_results)
                        if not next_liq_pool: continue

                        setups.append({
                            "poi": ob,
                            "direction": 'buy' if structure.direction == 1 else 'sell',
                            "target_liquidity": next_liq_pool.price,
                            "type": "Liquidity-Based (Sweep)",
                            "reasoning": [f"Liquidity Sweep followed by CHoCH at {structure.pivot.price:.5f}", f"POI: Extreme OB at {ob.low:.5f}-{ob.high:.5f}"]
                        })
                        break
                        
        return setups

    def _assemble_signal(self, trade: Dict, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        ob: OrderBlock = trade['poi']
        direction = trade['direction']
        
        entry_price = ob.open

        try:
            atr_at_ob = df['atr'].iloc[ob.bar_index]
            
            # --- Guard Clause to prevent NaN Stop Loss ---
            if pd.isna(atr_at_ob):
                logger.warning(f"[{symbol}] Skipping trade due to NaN ATR value at OB index {ob.bar_index}. OB is likely too old in the data series.")
                return None

            sl_buffer = self.params['stop_loss_atr_multiplier'] * atr_at_ob
        except (IndexError, TypeError, AttributeError):
             # Fallback if ATR series isn't available for some reason
            logger.error(f"[{symbol}] Could not calculate SL buffer due to an error. Skipping trade.")
            return None

        if direction == 'buy':
            stop_loss = ob.low - sl_buffer
        else:
            stop_loss = ob.high + sl_buffer
            
        take_profit = trade['target_liquidity']

        if stop_loss == entry_price: return None
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk == 0: return None
        
        rr_ratio = reward / risk
        if rr_ratio < self.params['min_risk_reward']:
            logger.debug(f"[{symbol}] Skipping trade. R:R of {rr_ratio:.2f} is below minimum of {self.params['min_risk_reward']:.2f}")
            return None

        signal_dict = {
            "symbol": symbol, "direction": direction, "entry_price": entry_price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "timeframe": self.primary_timeframe, "strategy_name": self.name,
            "confidence": 0.85 if self.params['require_fvg_confluence'] else 0.7,
            "description": f"{trade['type']} signal identified.",
            "detailed_reasoning": trade['reasoning'], "pattern": ob.ob_type,
            "signal_timestamp": str(ob.time),
        }
        return signal_dict

    def _has_fvg_confluence(self, ob: OrderBlock, smc_results: Dict) -> bool:
        """Checks if an Order Block is associated with a nearby Fair Value Gap."""
        for fvg in smc_results.get('fair_value_gaps', []):
            # FVG must be in the same direction as the intended move (opposite of OB)
            if fvg.direction == -ob.direction:
                # Check if the FVG is "close" to the OB (e.g., within 5 bars)
                if abs(fvg.bar_index - ob.bar_index) < 5:
                    return True
        return False
        
    def _check_if_choch_is_sweep(self, choch: Structure, analyzer: SmartMoneyConcepts) -> bool:
        """Determines if a CHoCH was likely caused by a liquidity sweep."""
        # A simple check: did the pivot that was broken by the CHoCH take out a *previous* major pivot?
        if choch.direction == 1: # Bullish CHoCH, broke a high
            # Was the low before this move lower than the previous major low?
            return analyzer.swing_low.price < analyzer.prev_swing_low.price
        else: # Bearish CHoCH, broke a low
            # Was the high before this move higher than the previous major high?
            return analyzer.swing_high.price > analyzer.prev_swing_high.price

    def _find_next_liquidity_target(self, direction: int, start_index: int, smc_results: Dict) -> Optional[Liquidity]:
        """Finds the next unmitigated liquidity pool to use as a target."""
        target_direction = 1 if direction == 1 else -1 # Buyside for longs, Sellside for shorts
        
        potential_targets = [
            pool for pool in smc_results.get('liquidity_pools', []) 
            if pool.direction == target_direction and pool.bar_index > start_index
        ]
        
        if not potential_targets:
            return None
        
        # Return the closest one
        return min(potential_targets, key=lambda x: x.bar_index)


# ==============================================================================
# --- Example Usage (for testing purposes) ---
# ==============================================================================
if __name__ == '__main__':
    # This block allows you to test the strategy in isolation.
    # You would need to provide a sample DataFrame.
    
    logger.add("smc_strategy_test.log") # Log to a file for easy debugging

    # 1. Create a sample DataFrame (e.g., load from a CSV file)
    # This should have columns: ['time', 'open', 'high', 'low', 'close', 'volume']
    # and a DatetimeIndex.
    try:
        # Assuming you have a file named 'test_data_M15.csv'
        sample_df = pd.read_csv('test_data_M15.csv', index_col='time', parse_dates=True)
        logger.info(f"Loaded sample data with {len(sample_df)} rows.")
    except FileNotFoundError:
        logger.error("Please create a 'test_data_M15.csv' file for testing.")
        sample_df = pd.DataFrame()

    if not sample_df.empty:
        # 2. Instantiate the strategy
        strategy = SMCStrategy(primary_timeframe="M15")

        # 3. Create the market_data structure the bot would provide
        market_data_packet = {
            "EUR/USD": {
                "M15": sample_df
            }
        }

        # 4. Generate signals
        # In a real bot, this would be in an async loop.
        import asyncio
        signals_found = asyncio.run(strategy.generate_signals(market_data=market_data_packet))
        
        # 5. Print results
        if signals_found:
            logger.info(f"🎉 Found {len(signals_found)} signals!")
            for sig in signals_found:
                logger.info(f" -> Signal: {sig['symbol']} {sig['direction']} @ {sig['entry_price']:.5f} | "
                            f"SL: {sig['stop_loss']:.5f} | TP: {sig['take_profit']:.5f}")
                logger.info(f"    Reason: {sig['detailed_reasoning']}")
        else:
            logger.info("No signals generated on the provided data.")