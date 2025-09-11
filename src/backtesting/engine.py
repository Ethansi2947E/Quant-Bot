'''
Backtesting Engine

This module contains the core Backtester class that orchestrates the entire
backtesting process.
'''

import pandas as pd
from loguru import logger
from typing import Type, Dict, Any, List
import asyncio

from src.trading_bot import SignalGenerator
from src.backtesting.risk_simulator import BacktestRiskManager
from src.backtesting.trade_simulator import SimulatedTrade
from src.backtesting.performance import PerformanceReport

class Backtester:
    """Orchestrates the backtesting process from start to finish."""

    def __init__(self, strategy_class: Type[SignalGenerator], data: pd.DataFrame, symbol: str,
                 initial_balance: float = 10000.0, strategy_params: Dict[str, Any] = None):
        """
        Initializes the Backtester.

        Args:
            strategy_class (Type[SignalGenerator]): The strategy class to be tested.
            data (pd.DataFrame): The historical OHLCV data.
            symbol (str): The symbol being tested (e.g., 'XAUUSD').
            initial_balance (float): The starting account balance for the simulation.
            strategy_params (Dict[str, Any]): Parameters to initialize the strategy with.
        """
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params if strategy_params is not None else {}
        self.data = data
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        self.strategy: SignalGenerator = None
        self.risk_manager = BacktestRiskManager() # Using the simplified backtest risk manager
        
        self.open_trades: List[SimulatedTrade] = []
        self.closed_trades: List[SimulatedTrade] = []
        self.equity_curve = pd.Series(dtype=float)

    async def run(self):
        """Runs the backtest from start to finish."""
        logger.info(f"Starting backtest for strategy: {self.strategy_class.__name__} on {self.symbol}")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        logger.info(f"Data range: {self.data.index[0]} to {self.data.index[-1]}")

        # 1. Instantiate the strategy
        self.strategy = self.strategy_class(**self.strategy_params)
        
        # 2. Prepare data by calculating all indicators upfront
        market_data_to_prepare = {self.symbol: {self.strategy.primary_timeframe: self.data}}
        prepared_market_data = self.strategy.prepare_data(market_data_to_prepare)
        prepared_data = prepared_market_data[self.symbol][self.strategy.primary_timeframe]

        lookback = len(self.data) - len(prepared_data) # Infer lookback from dropped NaN rows

        # 3. Main backtesting loop
        for i in range(len(prepared_data)):
            current_candle = prepared_data.iloc[i]
            # Pass an expanding slice of PREPARED data to the strategy
            historical_slice = prepared_data.iloc[0:i+1]

            market_data = {
                self.symbol: {
                    self.strategy.primary_timeframe: historical_slice
                }
            }

            # a. Check for trade closures
            self._check_closures(current_candle)

            # b. Generate new signals from prepared data
            signals = await self.strategy.generate_signals(market_data)

            # c. Process new signals
            if signals:
                self._process_signal(signals[0], current_candle)
            
            # d. Update equity
            self._update_equity(current_candle)

        # 4. Close any remaining open trades
        self._close_all_remaining_trades(prepared_data.iloc[-1])

        # 5. Generate and display performance report
        self.generate_and_display_report()

    def _check_closures(self, candle: pd.Series):
        """Check if any open trades should be closed based on the current candle."""
        trades_to_close = []
        for trade in self.open_trades:
            if trade.direction == 'buy':
                if candle['low'] <= trade.stop_loss:
                    trade.close(candle.name, trade.stop_loss, 'SL')
                    trades_to_close.append(trade)
                elif candle['high'] >= trade.take_profit:
                    trade.close(candle.name, trade.take_profit, 'TP')
                    trades_to_close.append(trade)
            elif trade.direction == 'sell':
                if candle['high'] >= trade.stop_loss:
                    trade.close(candle.name, trade.stop_loss, 'SL')
                    trades_to_close.append(trade)
                elif candle['low'] <= trade.take_profit:
                    trade.close(candle.name, trade.take_profit, 'TP')
                    trades_to_close.append(trade)
        
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            self.balance += trade.pnl
            logger.info(f"Closed {trade.direction} trade for {trade.symbol}. Reason: {trade.exit_reason}. PnL: {trade.pnl:.2f}")

    def _process_signal(self, signal: Dict[str, Any], candle: pd.Series):
        """Processes a new signal to open a trade."""
        if self.open_trades:
            return

        size = self.risk_manager.calculate_position_size(
            account_balance=self.balance,
            entry_price=signal['entry_price'],
            stop_loss_price=signal['stop_loss']
        )

        if size <= 0:
            return

        trade = SimulatedTrade(
            entry_price=signal['entry_price'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            position_size=size,
            direction=signal['direction'],
            symbol=signal['symbol'],
            entry_time=candle.name
        )
        self.open_trades.append(trade)
        logger.info(f"Opened new {trade.direction} trade for {trade.symbol} at {trade.entry_price:.5f} with size {trade.position_size}")

    def _update_equity(self, candle: pd.Series):
        """Updates the equity curve at each step."""
        current_equity = self.balance
        for trade in self.open_trades:
            price_change = candle['close'] - trade.entry_price
            if trade.direction == 'buy':
                current_equity += price_change * trade.position_size
            else:
                current_equity -= price_change * trade.position_size
        
        new_equity_point = pd.Series([current_equity], index=[candle.name])
        self.equity_curve = pd.concat([self.equity_curve, new_equity_point])

    def _close_all_remaining_trades(self, last_candle: pd.Series):
        """Close all trades that are still open at the end of the data."""
        if not self.open_trades:
            return
        
        for trade in self.open_trades:
            trade.close(last_candle.name, last_candle['close'], 'end_of_data')
            self.closed_trades.append(trade)
            self.balance += trade.pnl
        self.open_trades.clear()
        logger.info(f"Closed {len(self.closed_trades) - len(self.open_trades)} remaining trades at end of data.")

    def generate_and_display_report(self):
        """Generates and displays the final performance report."""
        report = PerformanceReport(self.closed_trades, self.equity_curve, self.initial_balance)
        report.generate_report()
        report.display_report()
        report.plot_equity_curve()
