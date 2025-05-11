"""
Price Action Support/Resistance Strategy

A fully rules-based price-action strategy using S/R zones, candlestick confirmation, wick rejection, and volume spikes.
Implements the core logic from Chapters 3–6 of the referenced PDF, with simple risk management.

Features:
- Pivot-based S/R zone detection (parameterized)
- Candlestick pattern and wick rejection confirmation
- Volume spike filter
- Parameterized for M15 by default, but supports M5/H1
- Signal output compatible with signal_processor.py
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager

class PriceActionSRStrategy(SignalGenerator):
    """
    Price Action S/R Strategy: Generates signals based on S/R zones, candlestick pattern, wick rejection, and volume spike.
    Parameterized for timeframe and all key thresholds.
    """
    def __init__(
        self,
        primary_timeframe: str = "M15",
        pivot_window: int = 10,
        wick_threshold: float = 0.5,
        volume_multiplier: float = 1.3,
        max_zones: int = 2,
        risk_per_trade: float = 0.01,
        **kwargs
    ):
        """
        Initialize the PriceActionSRStrategy.

        Args:
            primary_timeframe (str): Timeframe for analysis (default M15)
            pivot_window (int): Window for pivot high/low detection
            wick_threshold (float): Wick % threshold for rejection
            volume_multiplier (float): Volume spike threshold
            max_zones (int): Number of S/R zones to keep
            risk_per_trade (float): Fraction of equity to risk per trade
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.logger = logger
        self.name = "PriceActionSRStrategy"
        self.description = "Rules-based price action S/R strategy"
        self.version = "1.0.0"
        self.primary_timeframe = primary_timeframe
        self.pivot_window = pivot_window
        self.wick_threshold = wick_threshold
        self.volume_multiplier = volume_multiplier
        self.max_zones = max_zones
        self.risk_per_trade = risk_per_trade
        self.required_timeframes = [primary_timeframe]
        # Set lookback period to ensure we have enough data for analysis
        # We need at least pivot_window * 2 + 21 bars for complete analysis
        self.lookback_period = max(100, pivot_window * 2 + 50)  # Default 100 bars but more if needed
        self.lookback_periods = {primary_timeframe: self.lookback_period}
        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, 'get_instance') else RiskManager()
        
        # State tracking to prevent signal duplication
        self.processed_bars = {}  # {(symbol, timeframe): last_processed_timestamp}
        self.processed_zones = {}  # {(symbol, zone_type, zone_price): last_processed_timestamp}
        self.signal_cooldown = 86400  # 24 hours in seconds - don't reuse the same zone for this duration
        
        # For future extension: allow parameterization for M5/H1
        self.params = kwargs

    async def initialize(self):
        """
        Initialize the strategy with any necessary setup.
        Called by TradingBot during startup.
        """
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # Set up required timeframes (allow primary or parameters to override)
        timeframe_map = {
            "M5": 5,     # 5-minute bars
            "M15": 15,   # 15-minute bars
            "H1": 60     # 1-hour bars
        }
        
        # Log important parameters
        logger.info(f"Primary timeframe: {self.primary_timeframe}")
        logger.info(f"Pivot window: {self.pivot_window}")
        logger.info(f"Wick threshold: {self.wick_threshold}")
        logger.info(f"Volume multiplier: {self.volume_multiplier}")
        
        # Configure internal parameters based on timeframe
        if self.primary_timeframe == "M5":
            # For M5, use a larger pivot window (30 bars) to approximate the same lookback depth
            self.pivot_window = self.params.get("pivot_window_m5", 30)
            # Higher volume multiplier for faster timeframe
            self.volume_multiplier = self.params.get("volume_multiplier_m5", 1.2)
        elif self.primary_timeframe == "H1":
            # For H1, use smaller pivot window (5 bars)
            self.pivot_window = self.params.get("pivot_window_h1", 5)
            # Smaller wick threshold for H1
            self.wick_threshold = self.params.get("wick_threshold_h1", 0.4)
        
        # Ensure required_timeframes is set correctly
        self.required_timeframes = [self.primary_timeframe]
        
        logger.info(f"{self.name} initialization complete")
        return True

    def _find_pivot_highs(self, df: pd.DataFrame) -> List[float]:
        """
        Find pivot highs over the configured window.
        Args:
            df (pd.DataFrame): Price data with 'high' column
        Returns:
            List[float]: List of pivot high prices
        """
        pivots = []
        w = self.pivot_window
        for i in range(w, len(df) - w):
            window = df['high'].iloc[i-w:i+w+1]
            if df['high'].iloc[i] == window.max():
                pivots.append(df['high'].iloc[i])
        return pivots

    def _find_pivot_lows(self, df: pd.DataFrame) -> List[float]:
        """
        Find pivot lows over the configured window.
        Args:
            df (pd.DataFrame): Price data with 'low' column
        Returns:
            List[float]: List of pivot low prices
        """
        pivots = []
        w = self.pivot_window
        for i in range(w, len(df) - w):
            window = df['low'].iloc[i-w:i+w+1]
            if df['low'].iloc[i] == window.min():
                pivots.append(df['low'].iloc[i])
        return pivots

    def _cluster_levels(self, levels: List[float], tol: float = 0.003) -> List[float]:
        """
        Cluster price levels into horizontal zones within ±tol (as a fraction of price).
        Args:
            levels (List[float]): List of price levels
            tol (float): Tolerance as a fraction of price (default 0.003 = 0.3%)
        Returns:
            List[float]: Clustered zone center prices
        """
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current = [levels[0]]
        for price in levels[1:]:
            if abs(price - current[-1]) <= current[-1] * tol:
                current.append(price)
            else:
                clusters.append(current)
                current = [price]
        clusters.append(current)
        # Use mean of each cluster as zone center
        return [float(np.mean(cluster)) for cluster in clusters]

    def get_sr_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Compute and return the top N support and resistance zones.
        Args:
            df (pd.DataFrame): Price data with 'high' and 'low' columns
        Returns:
            Dict[str, List[float]]: {'support': [..], 'resistance': [..]}
        """
        highs = self._find_pivot_highs(df)
        lows = self._find_pivot_lows(df)
        res_zones = self._cluster_levels(highs, tol=0.003)
        sup_zones = self._cluster_levels(lows, tol=0.003)
        # Count touches for each zone
        def count_touches(prices, zones):
            return [sum(abs(p - z) <= z * 0.003 for p in prices) for z in zones]
        res_counts = count_touches(highs, res_zones)
        sup_counts = count_touches(lows, sup_zones)
        # Select top N by touches
        top_res = [z for _, z in sorted(zip(res_counts, res_zones), reverse=True)[:self.max_zones]]
        top_sup = [z for _, z in sorted(zip(sup_counts, sup_zones), reverse=True)[:self.max_zones]]
        return {'support': top_sup, 'resistance': top_res}

    def _is_in_zone(self, price: float, zone: float, tol: float = 0.003) -> bool:
        """Check if price is inside a zone (±tol)."""
        return abs(price - zone) <= zone * tol

    def _is_bullish_engulfing(self, prev, curr) -> bool:
        return (
            prev['close'] < prev['open'] and
            curr['close'] > curr['open'] and
            curr['close'] > prev['open'] and
            curr['open'] < prev['close']
        )

    def _is_bearish_engulfing(self, prev, curr) -> bool:
        return (
            prev['close'] > prev['open'] and
            curr['close'] < curr['open'] and
            curr['open'] > prev['close'] and
            curr['close'] < prev['open']
        )

    def _is_hammer(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        return (
            total > 0 and
            body / total < 0.3 and
            lower_wick > 2 * body and
            upper_wick < body
        )

    def _is_shooting_star(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        return (
            total > 0 and
            body / total < 0.3 and
            upper_wick > 2 * body and
            lower_wick < body
        )

    def _wick_rejection(self, candle, direction: str) -> bool:
        total = candle['high'] - candle['low']
        if total == 0:
            return False
        if direction == 'buy':
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            return lower_wick / total >= self.wick_threshold
        else:
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            return upper_wick / total >= self.wick_threshold

    def _volume_spike(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 20:
            return False
        avg_vol = df['tick_volume'].iloc[idx-20:idx].mean()
        return df['tick_volume'].iloc[idx] >= self.volume_multiplier * avg_vol

    def _pattern_match(self, df: pd.DataFrame, idx: int, direction: str) -> Optional[str]:
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        if direction == 'buy':
            if self._is_bullish_engulfing(prev, curr):
                return 'Bullish Engulfing'
            if self._is_hammer(curr):
                return 'Hammer'
            # Pin-bar: hammer with long lower wick
            if self._wick_rejection(curr, 'buy') and abs(curr['close'] - curr['open']) < (curr['high'] - curr['low']) * 0.3:
                return 'Pin-bar'
        else:
            if self._is_bearish_engulfing(prev, curr):
                return 'Bearish Engulfing'
            if self._is_shooting_star(curr):
                return 'Shooting Star'
            # Pin-bar: shooting star with long upper wick
            if self._wick_rejection(curr, 'sell') and abs(curr['close'] - curr['open']) < (curr['high'] - curr['low']) * 0.3:
                return 'Pin-bar'
        return None

    def _log_debug_info(self, symbol: str, df: pd.DataFrame, zones: Dict, signals: List[Dict]) -> None:
        """
        Log detailed debug information about S/R zones and signals.
        
        Args:
            symbol: Trading symbol
            df: Price data
            zones: Support/resistance zones
            signals: Generated signals
        """
        # Log S/R zones
        support_zones = zones.get('support', [])
        resistance_zones = zones.get('resistance', [])
        
        logger.info(f"[{symbol}] Analysis on {len(df)} candles, primary timeframe: {self.primary_timeframe}")
        logger.info(f"[{symbol}] Current price: {df['close'].iloc[-1]}")
        
        # Log support zones
        if support_zones:
            support_str = ", ".join([f"{z:.5f}" for z in support_zones])
            logger.info(f"[{symbol}] Support zones: {support_str}")
        else:
            logger.info(f"[{symbol}] No support zones detected")
            
        # Log resistance zones
        if resistance_zones:
            resistance_str = ", ".join([f"{z:.5f}" for z in resistance_zones])
            logger.info(f"[{symbol}] Resistance zones: {resistance_str}")
        else:
            logger.info(f"[{symbol}] No resistance zones detected")
        
        # Log signals
        if signals:
            for i, signal in enumerate(signals):
                direction = signal.get('direction', 'unknown')
                pattern = signal.get('pattern', 'unknown')
                entry = signal.get('entry_price', 0)
                sl = signal.get('stop_loss', 0)
                tp = signal.get('take_profit', 0)
                rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                
                logger.info(f"[{symbol}] Signal #{i+1}: {direction.upper()} - {pattern}")
                logger.info(f"[{symbol}] Entry: {entry:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, R:R = {rr:.2f}")
        else:
            logger.info(f"[{symbol}] No signals generated")

    def _rank_patterns(self, pattern: str) -> int:
        """
        Rank candlestick patterns by reliability.
        Higher number = more reliable pattern.
        
        Args:
            pattern: Candlestick pattern name
            
        Returns:
            int: Pattern rank (1-5)
        """
        pattern_ranks = {
            "Bullish Engulfing": 5,
            "Bearish Engulfing": 5,
            "Hammer": 4,
            "Shooting Star": 4,
            "Pin-bar": 3,
            # Add more patterns with their ranks as needed
        }
        return pattern_ranks.get(pattern, 1)  # Default rank = 1
        
    def _prioritize_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Prioritize signals when there are conflicts for the same symbol.
        Returns only the highest-priority signal for each symbol.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            List[Dict]: Prioritized signals (one per symbol)
        """
        if not signals:
            return []
            
        # Group signals by symbol
        signals_by_symbol = {}
        for signal in signals:
            symbol = signal.get('symbol')
            if symbol not in signals_by_symbol:
                signals_by_symbol[symbol] = []
            signals_by_symbol[symbol].append(signal)
            
        # For each symbol, if there's more than one signal, select the best one
        prioritized_signals = []
        
        for symbol, symbol_signals in signals_by_symbol.items():
            if len(symbol_signals) == 1:
                # Only one signal for this symbol, use it
                prioritized_signals.append(symbol_signals[0])
                continue
                
            logger.warning(f"Found {len(symbol_signals)} conflicting signals for {symbol}, prioritizing best one")
            
            # Score each signal based on multiple factors
            for signal in symbol_signals:
                score = 0.0
                
                # Factor 1: Risk-Reward ratio (higher is better)
                risk_reward = signal.get('risk_reward', 0)
                score += min(risk_reward * 10, 50)  # Cap at 50 points
                
                # Factor 2: Pattern strength (more reliable patterns score higher)
                pattern = signal.get('pattern', '')
                pattern_score = self._rank_patterns(pattern)
                score += pattern_score * 10  # Up to 50 points
                
                # Factor 3: Volume strength (higher volume spike is better)
                volume_score = signal.get('volume_score', 0)
                score += volume_score * 20  # Up to 20 points

                # Factor 4: Zone strength (how many times price has respected this zone)
                zone_strength = signal.get('zone_touches', 1)
                score += min(zone_strength * 5, 20)  # Up to 20 points
                
                # Store the score in the signal
                signal['priority_score'] = score
                
                logger.info(f"Signal {signal.get('direction')} {pattern} scored {score:.1f} points")
                
            # Sort signals by score (descending) and take the top one
            best_signal = sorted(symbol_signals, key=lambda s: s.get('priority_score', 0), reverse=True)[0]
            logger.info(f"Selected {best_signal.get('direction')} {best_signal.get('pattern')} as best signal for {symbol} with score {best_signal.get('priority_score', 0):.1f}")
            
            prioritized_signals.append(best_signal)
            
        return prioritized_signals

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        """
        Generate trading signals based on S/R zone entry, candlestick pattern, wick rejection, and volume spike.
        Only processes new bars and zones that haven't been used recently.
        
        Args:
            market_data (Dict[str, Any]): Dict of symbol to DataFrame or dict of timeframes
            symbol (str, optional): Symbol to process (if only one)
        Returns:
            List[Dict]: List of signal dicts ready for signal_processor
        """
        signals = []
        current_time = datetime.now().timestamp()
        symbols = [symbol] if symbol else list(market_data.keys())
        
        for sym in symbols:
            logger.info(f"Analyzing {sym} with {self.name}")
            df = market_data[sym]
            if isinstance(df, dict):
                df = df.get(self.primary_timeframe)
            if df is None or len(df) < self.pivot_window + 21:
                logger.warning(f"Insufficient data for {sym}: Need at least {self.pivot_window + 21} bars, got {len(df) if df is not None else 0}")
                continue
                
            # Check if we've already processed this bar for this symbol and timeframe
            bar_key = (sym, self.primary_timeframe)
            last_timestamp = None
            
            try:
                last_timestamp = pd.to_datetime(df.index[-1])
                last_timestamp_str = str(last_timestamp)
            except:
                logger.warning(f"Could not extract timestamp from dataframe for {sym}")
                last_timestamp_str = str(current_time)
                
            if bar_key in self.processed_bars and self.processed_bars[bar_key] == last_timestamp_str:
                logger.debug(f"Already processed latest bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")
                continue
                
            # We're processing a new bar, update our tracking
            self.processed_bars[bar_key] = last_timestamp_str
            logger.debug(f"Processing new bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")
            
            df = df.copy()
            if 'tick_volume' not in df.columns:
                df['tick_volume'] = df.get('volume', 1)
                
            # Get support/resistance zones
            zones = self.get_sr_zones(df)
            symbol_signals = []
            
            # Process support and resistance zones
            for direction, zone_list in [('buy', zones['support']), ('sell', zones['resistance'])]:
                zone_type = 'support' if direction == 'buy' else 'resistance'
                logger.debug(f"Checking {len(zone_list)} {zone_type} zones for {sym}")
                
                for zone in zone_list:
                    # Check if we've already used this zone recently
                    zone_key = (sym, zone_type, round(zone, 5))
                    if zone_key in self.processed_zones:
                        last_used_time = self.processed_zones[zone_key]
                        time_since_use = current_time - last_used_time
                        
                        if time_since_use < self.signal_cooldown:
                            cooldown_remaining = self.signal_cooldown - time_since_use
                            hours_remaining = cooldown_remaining / 3600
                            logger.debug(f"Skipping {zone_type} zone {zone:.5f} - on cooldown for {hours_remaining:.1f} more hours")
                            continue
                    
                    # Count zone touches for strength
                    zone_touches = sum(1 for i in range(len(df)) if self._is_in_zone(df['close'].iloc[i], zone))
                    
                    # We only care about the most recent bar that hasn't been processed yet
                    # This ensures we only generate signals for new price action
                    idx = len(df) - 1
                    process_idx = idx - 1  # We use previous completed bar for pattern recognition
                    
                    if process_idx < self.pivot_window + 20:
                        logger.debug(f"Not enough bars to check for patterns at index {process_idx}")
                        continue
                        
                    price = df['close'].iloc[process_idx]
                    if not self._is_in_zone(price, zone):
                        continue
                        
                    logger.debug(f"Price {price:.5f} is in {zone_type} zone {zone:.5f}")
                    
                    # Check pattern
                    pattern = self._pattern_match(df, process_idx, direction)
                    if not pattern:
                        logger.debug(f"No {direction} pattern detected")
                        continue
                    logger.debug(f"Pattern match: {pattern}")
                    
                    # Check wick rejection
                    if not self._wick_rejection(df.iloc[process_idx], direction):
                        logger.debug(f"No wick rejection")
                        continue
                    logger.debug(f"Wick rejection confirmed")
                    
                    # Check volume spike and calculate volume score
                    if not self._volume_spike(df, process_idx):
                        logger.debug(f"No volume spike")
                        continue
                    logger.debug(f"Volume spike confirmed")
                    
                    # Calculate volume score (how much above threshold)
                    avg_vol = df['tick_volume'].iloc[process_idx-20:process_idx].mean()
                    current_vol = df['tick_volume'].iloc[process_idx]
                    volume_score = min((current_vol / avg_vol) / self.volume_multiplier, 2.0)  # Cap at 2.0
                    
                    # All conditions met - create signal
                    entry = df['open'].iloc[idx] if idx < len(df) else price
                    stop = df['low'].iloc[process_idx] if direction == 'buy' else df['high'].iloc[process_idx]
                    
                    # Find next opposite zone for TP
                    opp_zones = zones['resistance'] if direction == 'buy' else zones['support']
                    tp = None
                    for opp in (sorted(opp_zones) if direction == 'buy' else sorted(opp_zones, reverse=True)):
                        if (direction == 'buy' and opp > entry) or (direction == 'sell' and opp < entry):
                            tp = opp
                            break
                    
                    # Fallback TP calculation if no opposite zone
                    if tp is None:
                        risk = abs(entry - stop)
                        tp = entry + 2 * risk if direction == 'buy' else entry - 2 * risk
                        logger.debug(f"Using fallback 2R TP: {tp:.5f}")
                    
                    # Position sizing
                    equity = self.risk_manager.get_account_balance() if hasattr(self.risk_manager, 'get_account_balance') else 10000
                    size = (equity * self.risk_per_trade) / max(abs(entry - stop), 1e-6)
                    
                    # Calculate risk reward ratio
                    risk_reward = abs(tp - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
                    
                    # Confidence: 1.0 if all confluences, else 0.8
                    confidence = 1.0
                    
                    # Enhanced detailed reason for better transparency
                    reason = (
                        f"{pattern} at {zone_type} zone {zone:.5f} with "
                        f"wick rejection and volume spike. "
                        f"Risk:Reward = {risk_reward:.2f}"
                    )
                    
                    signal = {
                        'symbol': sym,
                        'direction': direction,
                        'entry_price': float(entry),
                        'stop_loss': float(stop),
                        'take_profit': float(tp),
                        'confidence': confidence,
                        'size': float(size),
                        'timeframe': self.primary_timeframe,
                        'reason': reason,
                        'pattern': pattern,
                        'zone': float(zone),
                        'zone_touches': zone_touches,
                        'volume_score': volume_score,
                        'bar_index': int(process_idx),
                        'risk_reward': float(risk_reward),
                        'signal_timestamp': str(df.index[process_idx])
                    }
                    
                    logger.info(f"Signal generated for {sym} {direction.upper()}: {pattern} at {zone_type} zone")
                    
                    # Mark this zone as processed with the current timestamp
                    self.processed_zones[zone_key] = current_time
                    
                    symbol_signals.append(signal)
                    
                    # Only generate one signal per zone, then move to next zone
                    break
            
            # Prioritize signals if there are conflicting ones
            if len(symbol_signals) > 1:
                # Store all signals for logging purposes
                all_signals = symbol_signals.copy()
                
                # Apply prioritization
                symbol_signals = self._prioritize_signals(symbol_signals)
                
                logger.info(f"Prioritized {len(all_signals)} signals down to {len(symbol_signals)} for {sym}")
            
            # Add symbol's signals to overall signals list
            signals.extend(symbol_signals)
            
            # Log debug information 
            self._log_debug_info(sym, df, zones, symbol_signals)
        
        # Cleanup old processed zones entries (older than 48 hours)
        cleanup_time = current_time - (self.signal_cooldown * 2)
        old_keys = [k for k, v in self.processed_zones.items() if v < cleanup_time]
        for k in old_keys:
            del self.processed_zones[k]
            
        if old_keys:
            logger.debug(f"Cleaned up {len(old_keys)} old zone records")
            
        return signals 