'''
Backtesting Performance Reporter

This module provides a class to calculate and display key performance indicators (KPIs)
from the results of a backtest.
'''

import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger

from src.backtesting.trade_simulator import SimulatedTrade


class PerformanceReport:
    """Calculates and displays a performance report from a list of closed trades."""

    def __init__(self, closed_trades: List[SimulatedTrade], equity_curve: pd.Series, initial_balance: float):
        if not closed_trades:
            logger.warning("PerformanceReport initialized with no trades.")
            self.trades_df = pd.DataFrame()
        else:
            self.trades_df = pd.DataFrame([t.__dict__ for t in closed_trades])
            if 'exit_time' in self.trades_df.columns:
                self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
                self.trades_df.set_index('exit_time', inplace=True, drop=False)

        self.equity_curve = equity_curve
        self.initial_balance = initial_balance
        self.metrics = {}

    def generate_report(self):
        """Calculate all performance metrics."""
        if self.trades_df.empty:
            self.metrics['error'] = "No trades to analyze."
            return

        self._calculate_core_metrics()
        self._calculate_drawdown_metrics()
        self._calculate_advanced_metrics()
        self._calculate_streaks()

    def _calculate_core_metrics(self):
        total_trades = len(self.trades_df)
        self.metrics['total_trades'] = total_trades
        self.metrics['net_pnl'] = self.trades_df['pnl'].sum()
        self.metrics['gross_profit'] = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        self.metrics['gross_loss'] = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum()
        winning_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['pnl'] < 0])
        self.metrics['winning_trades'] = winning_trades
        self.metrics['losing_trades'] = losing_trades
        self.metrics['win_rate'] = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        self.metrics['avg_trade_pnl'] = self.metrics['net_pnl'] / total_trades if total_trades > 0 else 0
        self.metrics['avg_win'] = self.metrics['gross_profit'] / winning_trades if winning_trades > 0 else 0
        self.metrics['avg_loss'] = self.metrics['gross_loss'] / losing_trades if losing_trades > 0 else 0
        gross_loss_abs = abs(self.metrics['gross_loss'])
        self.metrics['profit_factor'] = self.metrics['gross_profit'] / gross_loss_abs if gross_loss_abs > 0 else float('inf')

    def _calculate_drawdown_metrics(self):
        if self.equity_curve.empty:
            self.metrics['max_drawdown_pct'] = 0
            self.metrics['avg_drawdown_pct'] = 0
            return
        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = (self.equity_curve - peak) / peak
        self.metrics['max_drawdown_pct'] = abs(drawdown.min() * 100)
        self.metrics['avg_drawdown_pct'] = abs(drawdown[drawdown < 0].mean() * 100)

    def _calculate_advanced_metrics(self):
        # Assuming 252 trading days in a year for annualization
        trading_days_per_year = 252
        # Assuming 0% risk-free rate
        risk_free_rate = 0.0

        daily_returns = self.equity_curve.pct_change().dropna()
        
        if daily_returns.empty or len(daily_returns) < 2:
            self.metrics['sharpe_ratio'] = 0
            self.metrics['sortino_ratio'] = 0
            self.metrics['calmar_ratio'] = 0
            return

        # Sharpe Ratio
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        daily_sharpe = (avg_daily_return - risk_free_rate) / std_daily_return if std_daily_return != 0 else 0
        self.metrics['sharpe_ratio'] = daily_sharpe * np.sqrt(trading_days_per_year)

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std()
        daily_sortino = (avg_daily_return - risk_free_rate) / downside_std if downside_std != 0 else 0
        self.metrics['sortino_ratio'] = daily_sortino * np.sqrt(trading_days_per_year)

        # Calmar Ratio
        cumulative_return = (self.equity_curve.iloc[-1] / self.initial_balance) - 1
        annualized_return = (1 + cumulative_return) ** (trading_days_per_year / len(self.equity_curve)) - 1
        max_drawdown_decimal = self.metrics.get('max_drawdown_pct', 0) / 100
        self.metrics['calmar_ratio'] = annualized_return / max_drawdown_decimal if max_drawdown_decimal != 0 else 0

    def _calculate_streaks(self):
        wins = self.trades_df['pnl'] > 0
        losses = self.trades_df['pnl'] < 0
        
        win_streak = 0
        max_win_streak = 0
        loss_streak = 0
        max_loss_streak = 0

        for pnl in self.trades_df['pnl']:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
            elif pnl < 0:
                loss_streak += 1
                win_streak = 0
            else:
                win_streak = 0
                loss_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)

        self.metrics['longest_win_streak'] = max_win_streak
        self.metrics['longest_loss_streak'] = max_loss_streak

    def display_report(self):
        if not self.metrics or 'error' in self.metrics:
            logger.error(f"Could not generate report: {self.metrics.get('error', 'Unknown error')}")
            return

        report = f"""
        ======================================================
        |               Backtest Performance Report              |
        ======================================================

        --- Summary ---
        Total Trades:           {self.metrics.get('total_trades', 0)}
        Win Rate:               {self.metrics.get('win_rate', 0):.2f}%
        Profit Factor:          {self.metrics.get('profit_factor', 0):.2f}

        --- Profit & Loss ---
        Net PnL:                ${self.metrics.get('net_pnl', 0):.2f}
        Gross Profit:           ${self.metrics.get('gross_profit', 0):.2f}
        Gross Loss:             ${self.metrics.get('gross_loss', 0):.2f}

        --- Averages ---
        Avg. Trade PnL:         ${self.metrics.get('avg_trade_pnl', 0):.2f}
        Avg. Win:               ${self.metrics.get('avg_win', 0):.2f}
        Avg. Loss:              ${self.metrics.get('avg_loss', 0):.2f}

        --- Risk & Drawdown ---
        Max Drawdown:           {self.metrics.get('max_drawdown_pct', 0):.2f}%
        Avg. Drawdown:          {self.metrics.get('avg_drawdown_pct', 0):.2f}%

        --- Advanced Metrics ---
        Sharpe Ratio:           {self.metrics.get('sharpe_ratio', 0):.2f}
        Sortino Ratio:          {self.metrics.get('sortino_ratio', 0):.2f}
        Calmar Ratio:           {self.metrics.get('calmar_ratio', 0):.2f}

        --- Streaks ---
        Longest Win Streak:     {self.metrics.get('longest_win_streak', 0)}
        Longest Loss Streak:    {self.metrics.get('longest_loss_streak', 0)}

        ======================================================
        """
        print(report)

    def plot_equity_curve(self, save_path: str = 'equity_curve.png'):
        """Generates and saves a plot of the equity curve."""
        try:
            import matplotlib
            matplotlib.use('Agg') # Use a non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed. Cannot plot equity curve. Run: pip install matplotlib")
            return

        if self.equity_curve.empty:
            logger.warning("Equity curve is empty. Cannot plot.")
            return

        plt.figure(figsize=(12, 6))
        self.equity_curve.plot(title='Equity Curve', grid=True)
        plt.xlabel("Trade Number")
        plt.ylabel("Equity")
        plt.fill_between(self.equity_curve.index, self.equity_curve, self.initial_balance, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        logger.success(f"Equity curve plot saved to {save_path}")