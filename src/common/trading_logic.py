'''
This module provides TradingLogic, a class that encapsulates the core trading
logic used both in live trading bots and backtesting scenarios.
'''

import pandas as pd
from typing import List, Dict
from loguru import logger

from src.signal_generator import SignalGenerator
from src.market_analysis import MarketAnalysis
from src.risk_manager import RiskManager
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis


class TradingLogic:
    '''
    TradingLogic encapsulates the core trade decision-making process,
    including market analysis and trade simulation.
    '''
    def __init__(self, config: Dict):
        '''
        Initialize the TradingLogic with the required configuration.

        Args:
            config (Dict): Configuration dictionary.
        '''
        self.config = config
        self.signal_generator = SignalGenerator()
        self.market_analysis = MarketAnalysis()
        self.risk_manager = RiskManager()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()

    def analyze_market_data(self, data: pd.DataFrame, symbol: str,
                              timeframe: str) -> List[Dict]:
        '''
        Analyze market data and generate trade signals.

        Args:
            data (pd.DataFrame): Historical market data.
            symbol (str): Trading symbol.
            timeframe (str): Data timeframe.

        Returns:
            List[Dict]: List of generated signals.
        '''
        try:
            signals = []
            window_size = 100
            # Pre-calculate indicators for entire dataset.
            try:
                data = self.signal_generator.calculate_indicators(data.copy())
                data['atr'] = self.market_analysis.calculate_atr(data)
                data['volatility_state'] = self.market_analysis.classify_volatility(data['atr'])
                data['trend_state'] = self.market_analysis.classify_trend(data)
                data['structure'] = self.market_analysis.analyze_market_structure(df=data, timeframe=timeframe)
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
                return []

            # Process each candle in the dataset.
            for i in range(window_size, len(data)):
                try:
                    window_data = data.iloc[max(0, i - window_size):i + 1].copy()
                    current_candle = window_data.iloc[-1]
                    if signals and (current_candle.name - signals[-1]['timestamp']).total_seconds() < self.config.get('min_signal_spacing', 3600):
                        continue

                    market_conditions = {
                        'trend': current_candle['trend_state'],
                        'volatility': current_candle['volatility_state'],
                        'structure': current_candle['structure'],
                        'session': self._determine_session(current_candle.name),
                        'atr': current_candle['atr']
                    }
                    # Calculate required scores
                    structure_data = self.market_analysis.analyze_market_structure(window_data, timeframe)
                    volume_data = self.volume_analysis.analyze(window_data)
                    smc_data = self.smc_analysis.analyze(window_data)
                    
                    # Prepare data for MTF analysis
                    mtf_data = {timeframe: window_data}
                    mtf_analysis_result = self.mtf_analysis.analyze_mtf(mtf_data, timeframe)
                    
                    # Calculate individual scores
                    structure_score = self.signal_generator._calculate_structure_score(structure_data)
                    volume_score = self.signal_generator._calculate_volume_score(volume_data)
                    smc_score = self.signal_generator._calculate_smc_score(smc_data, window_data)
                    mtf_score = self.signal_generator._calculate_mtf_score(mtf_analysis_result)
                    
                    # Prepare market data with all required components
                    market_data = {
                        'data': window_data,
                        'market_conditions': market_conditions,
                        'structure_score': structure_score,
                        'volume_score': volume_score,
                        'smc_score': smc_score,
                        'mtf_score': mtf_score,
                        'mtf_confidence': mtf_analysis_result.get('confidence_factor', 1.0),
                        'is_ranging': market_conditions['trend'] == 'ranging',
                        'volatility_ratio': market_conditions['atr'] / window_data['atr'].mean()
                    }
                    
                    signal = self.signal_generator.generate_signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        market_data=market_data
                    )
                    if signal and signal.get('direction') and signal.get('direction') != 'HOLD':
                        signal['market_conditions'] = market_conditions
                        if self._validate_signal(signal, window_data):
                            signal['timestamp'] = current_candle.name
                            signals.append(signal)
                            logger.info(
                                f"Generated {signal['direction']} signal for {symbol} ({timeframe})"
                            )
                except Exception as e:
                    logger.warning(
                        f"Error processing candle at index {i} for {symbol} {timeframe}: {str(e)}"
                    )
                    continue
            logger.info(
                f"Analysis complete for {symbol} {timeframe}. Found {len(signals)} actionable signals."
            )
            return signals
        except Exception as e:
            logger.error(f"Error in market analysis for {symbol} {timeframe}: {str(e)}")
            return []

    def _determine_session(self, timestamp) -> str:
        '''
        Determine the trading session based on a timestamp.

        Args:
            timestamp: A datetime object.

        Returns:
            str: The session identifier.
        '''
        from datetime import time
        if time(9, 0) <= timestamp.time() <= time(16, 0):
            return "session_A"
        return "session_B"

    def _validate_signal(self, signal: Dict, data: pd.DataFrame) -> bool:
        '''
        Validate signal strength using RiskManager.

        Args:
            signal (Dict): A trade signal.
            data (pd.DataFrame): Data for validation.

        Returns:
            bool: True if valid, otherwise False.
        '''
        try:
            is_valid, reason = self.risk_manager.validate_trade(
                signal_type=signal['direction'],
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                confidence=signal.get('confidence', 0.5)
            )
            if not is_valid:
                logger.info(f"Signal rejected: {reason}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error in signal validation: {str(e)}")
            return False

    def simulate_trades(self, signals: List[Dict], data: pd.DataFrame,
                        symbol: str) -> List[Dict]:
        '''
        Simulate trade executions based on generated signals.

        Args:
            signals (List[Dict]): List of trade signals.
            data (pd.DataFrame): Historical market data.
            symbol (str): Trading symbol.

        Returns:
            List[Dict]: List of executed trades with details.
        '''
        trades = []
        trade_id = 1
        
        # Calculate ATR if not already present
        if 'atr' not in data.columns:
            try:
                # Calculate True Range
                high = data['high']
                low = data['low']
                close = data['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Calculate ATR with 14 period
                data['atr'] = tr.rolling(window=14).mean()
                data['atr'] = data['atr'].fillna(method='bfill')  # Fill any NaN values
                logger.info(f"ATR calculated successfully. Latest ATR: {data['atr'].iloc[-1]:.5f}")
            except Exception as e:
                logger.error(f"Error calculating ATR: {str(e)}")
                return []

        for signal in signals:
            try:
                if not all(k in signal for k in ['direction', 'entry_price']):
                    continue
                
                # Get data window for stop calculation
                signal_time = signal.get('timestamp')
                if signal_time:
                    data_window = data[data.index <= signal_time].copy()
                else:
                    data_window = data.copy()
                
                if len(data_window) < 14:  # Minimum required for ATR
                    logger.warning(f"Insufficient data for ATR calculation at signal {signal}")
                    continue

                stop_loss, take_profits = self.risk_manager.calculate_dynamic_stops(
                    df=data_window,
                    direction=signal['direction'],
                    entry_price=signal['entry_price'],
                    volatility_state=signal.get('market_conditions', {}).get('volatility', 'normal')
                )
                
                if stop_loss is None or not take_profits:
                    logger.warning(f"Could not calculate stops for signal {signal}")
                    continue

                position_size = self.risk_manager.calculate_dynamic_position_size(
                    account_balance=self.config["initial_balance"],
                    risk_amount=self.config["initial_balance"] * self.config.get("risk_per_trade", 0.01),
                    entry_price=signal['entry_price'],
                    stop_loss=stop_loss,
                    symbol=symbol,
                    market_condition=signal.get('market_conditions', {}).get('trend', 'ranging'),
                    volatility_state=signal.get('market_conditions', {}).get('volatility', 'normal'),
                    session=signal.get('market_conditions', {}).get('session', 'normal'),
                    correlation=0.0
                )
                
                trade = {
                    'id': trade_id,
                    'direction': signal['direction'],
                    'entry_time': signal.get('timestamp'),
                    'entry_price': signal['entry_price'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profits[0]['price'],
                    'position_size': position_size,
                    'symbol': symbol,
                    'timeframe': signal.get('timeframe', 'M15'),
                    'strategy': signal.get('strategy', 'default'),
                    'market_conditions': signal.get('market_conditions', {}),
                    'partial_take_profits': take_profits,
                    'risk_amount': self.config["initial_balance"] * self.config.get("risk_per_trade", 0.01),
                    'account_balance': self.config["initial_balance"]
                }
                
                future_data = data[data.index > trade['entry_time']]
                if not future_data.empty:
                    exit_details = self._simulate_trade_exit_with_partials(trade, future_data)
                    trade.update(exit_details)
                    if 'exit_time' in trade and 'exit_price' in trade:
                        trades.append(trade)
                        trade_id += 1
                        logger.info(f"Executed {trade['direction']} trade for {symbol}")
            except Exception as e:
                logger.error(f"Error simulating trade for signal {signal}: {str(e)}")
                continue
        return trades

    def _simulate_trade_exit_with_partials(self, trade: Dict,
                                           future_data: pd.DataFrame) -> Dict:
        '''
        Simulate trade exit logic with stop loss, take profit, and trailing stop conditions.

        Args:
            trade (Dict): The trade being simulated.
            future_data (pd.DataFrame): Data following the trade entry.

        Returns:
            Dict: Exit details for the trade.
        '''
        entry_price = trade['entry_price']
        initial_stop = trade['stop_loss']  # Store initial stop loss
        current_stop = initial_stop
        trade_type = trade['direction']
        position_size = trade['position_size']
        partial_take_profits = trade.get('partial_take_profits', [])
        partial_exits = []
        remaining_position = position_size
        
        # Calculate initial risk in pips
        initial_risk = abs(entry_price - initial_stop)
        
        max_adverse_excursion = 0
        max_favorable_excursion = 0
        
        # Calculate ATR if not present
        if 'atr' not in future_data.columns:
            high = future_data['high']
            low = future_data['low']
            close = future_data['close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            future_data['atr'] = tr.rolling(window=14).mean()
            future_data['atr'].fillna(method='bfill', inplace=True)

        for timestamp, row in future_data.iterrows():
            current_price = row['close']
            current_atr = row['atr']
            
            # Update MAE and MFE
            if trade_type.upper() == 'BUY':
                adverse_exc = (entry_price - row['low']) / entry_price * 100
                favorable_exc = (row['high'] - entry_price) / entry_price * 100
            else:
                adverse_exc = (row['high'] - entry_price) / entry_price * 100
                favorable_exc = (entry_price - row['low']) / entry_price * 100
            max_adverse_excursion = max(max_adverse_excursion, adverse_exc)
            max_favorable_excursion = max(max_favorable_excursion, favorable_exc)
            
            # Check for trailing stop adjustment
            should_adjust, new_stop = self.risk_manager.should_adjust_stops(
                trade={
                    'entry_price': entry_price,
                    'initial_stop': initial_stop,
                    'stop_loss': current_stop,
                    'direction': trade_type,
                    'partial_take_profits': partial_take_profits
                },
                current_price=current_price,
                current_atr=current_atr
            )
            
            if should_adjust:
                current_stop = new_stop
                logger.info(f"Trailing stop adjusted to {current_stop:.5f}")
            
            # Check for stop loss hit
            if trade_type.upper() == 'BUY' and row['low'] <= current_stop:
                # Calculate R-multiple (negative for loss)
                r_multiple = -(abs(entry_price - current_stop) / initial_risk)
                pnl = (current_stop - entry_price) * remaining_position * 100000
                return {
                    'exit_time': timestamp,
                    'exit_price': current_stop,
                    'exit_reason': 'Trailing Stop' if current_stop > initial_stop else 'Stop Loss',
                    'pnl': pnl,
                    'r_multiple': r_multiple,
                    'max_adverse_excursion': max_adverse_excursion,
                    'max_favorable_excursion': max_favorable_excursion,
                    'partial_exits': partial_exits,
                    'final_stop': current_stop
                }
            elif trade_type.upper() == 'SELL' and row['high'] >= current_stop:
                # Calculate R-multiple (negative for loss)
                r_multiple = -(abs(current_stop - entry_price) / initial_risk)
                pnl = (entry_price - current_stop) * remaining_position * 100000
                return {
                    'exit_time': timestamp,
                    'exit_price': current_stop,
                    'exit_reason': 'Trailing Stop' if current_stop < initial_stop else 'Stop Loss',
                    'pnl': pnl,
                    'r_multiple': r_multiple,
                    'max_adverse_excursion': max_adverse_excursion,
                    'max_favorable_excursion': max_favorable_excursion,
                    'partial_exits': partial_exits,
                    'final_stop': current_stop
                }
            
            # Check for partial take profits
            for tp in partial_take_profits[:]:  # Use slice copy to avoid modifying during iteration
                if trade_type.upper() == 'BUY' and row['high'] >= tp['price']:
                    exit_size = position_size * tp['size']
                    remaining_position -= exit_size
                    # Calculate R-multiple for partial exit
                    r_multiple = abs(tp['price'] - entry_price) / initial_risk
                    partial_pnl = (tp['price'] - entry_price) * exit_size * 100000
                    partial_exits.append({
                        'time': timestamp,
                        'price': tp['price'],
                        'size': exit_size,
                        'pnl': partial_pnl,
                        'r_multiple': r_multiple
                    })
                    partial_take_profits.remove(tp)  # Remove hit target
                elif trade_type.upper() == 'SELL' and row['low'] <= tp['price']:
                    exit_size = position_size * tp['size']
                    remaining_position -= exit_size
                    # Calculate R-multiple for partial exit
                    r_multiple = abs(entry_price - tp['price']) / initial_risk
                    partial_pnl = (entry_price - tp['price']) * exit_size * 100000
                    partial_exits.append({
                        'time': timestamp,
                        'price': tp['price'],
                        'size': exit_size,
                        'pnl': partial_pnl,
                        'r_multiple': r_multiple
                    })
                    partial_take_profits.remove(tp)  # Remove hit target
            
            # If all partials hit, close remaining position at current price
            if not partial_take_profits and remaining_position > 0:
                if trade_type.upper() == 'BUY':
                    r_multiple = abs(current_price - entry_price) / initial_risk
                    pnl = (current_price - entry_price) * remaining_position * 100000
                else:
                    r_multiple = abs(entry_price - current_price) / initial_risk
                    pnl = (entry_price - current_price) * remaining_position * 100000
                
                # Calculate total PnL including partials
                total_pnl = pnl + sum(p['pnl'] for p in partial_exits)
                
                return {
                    'exit_time': timestamp,
                    'exit_price': current_price,
                    'exit_reason': 'All Targets Hit',
                    'pnl': total_pnl,
                    'r_multiple': r_multiple,
                    'max_adverse_excursion': max_adverse_excursion,
                    'max_favorable_excursion': max_favorable_excursion,
                    'partial_exits': partial_exits,
                    'final_stop': current_stop
                }
        
        # If no exit condition met, close at last price
        last_price = future_data.iloc[-1]['close']
        if trade_type.upper() == 'BUY':
            r_multiple = abs(last_price - entry_price) / initial_risk
            pnl = (last_price - entry_price) * remaining_position * 100000
        else:
            r_multiple = abs(entry_price - last_price) / initial_risk
            pnl = (entry_price - last_price) * remaining_position * 100000
        
        # Calculate total PnL including partials
        total_pnl = pnl + sum(p['pnl'] for p in partial_exits)
        
        return {
            'exit_time': future_data.index[-1],
            'exit_price': last_price,
            'exit_reason': 'End of Data',
            'pnl': total_pnl,
            'r_multiple': r_multiple,
            'max_adverse_excursion': max_adverse_excursion,
            'max_favorable_excursion': max_favorable_excursion,
            'partial_exits': partial_exits,
            'final_stop': current_stop
        } 