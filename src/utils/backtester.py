"""
Backtester module for trading strategies.

Supports bar-by-bar simulation of strategies like BreakoutReversalStrategy and ConfluencePriceActionStrategy
on historical data from CSV or MT5, with multi-symbol and multi-timeframe support.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import os
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import json
from src.utils.performance_tracker import PerformanceTracker
from datetime import datetime
import matplotlib.dates as mdates
from src.mt5_handler import MT5Handler

class Backtester:
    """
    Core backtesting engine for trading strategies.
    
    Usage:
        - Initialize with strategy, data loader, config, and symbols
        - Call run() to execute the backtest
        - Access results and reports after completion
    
    Note: If 'profile': True is set in config, profiling will be enabled for strategy.generate_signals,
    and parallelization will be automatically disabled to avoid cProfile conflicts.
    """
    def __init__(self, 
                 strategy, 
                 data_loader: Callable, 
                 symbols: List[str],
                 timeframes: List[str],
                 config: Optional[Dict] = None,
                 risk_manager: Optional[Any] = None,
                 performance_tracker: Optional[Any] = None):
        """
        Args:
            strategy: An instance of a SignalGenerator-compatible strategy
            data_loader: Function to load historical data (CSV or MT5)
            symbols: List of symbols to backtest
            timeframes: List of timeframes required by the strategy
            config: Optional configuration dictionary
            risk_manager: Optional RiskManager instance for position sizing
            performance_tracker: Optional PerformanceTracker instance for advanced metrics
        """
        self.strategy = strategy
        self.data_loader = data_loader
        self.symbols = symbols
        self.timeframes = timeframes
        self.config = config or {}
        self.risk_manager = risk_manager
        self.performance_tracker = performance_tracker
        self.results = None
        self.trade_log = []
        self.equity_curve = []
        self.performance = None
        # TODO: Add more attributes as needed (risk manager, performance tracker, etc.)

    def load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load historical data for all symbols and timeframes.
        Returns:
            Nested dict: {symbol: {timeframe: DataFrame}}
        """
        # TODO: Implement data loading using self.data_loader
        raise NotImplementedError

    async def run(self):
        """
        Main backtesting loop. Simulates bar-by-bar trading for all symbols.
        Loads data, aligns by datetime, iterates through each bar, and simulates trades.
        Stores results in self.results, self.trade_log, and self.equity_curve.
        
        Note: If profiling is enabled, parallelization is automatically disabled by the strategy.
        """
        # Load data
        data = self.data_loader(self.symbols, self.timeframes)
        # Collect all unique datetimes across all symbols/timeframes
        all_datetimes = set()
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = data[symbol][tf]
                if not df.empty:
                    all_datetimes.update(df.index)
        # Sort all datetimes
        all_datetimes = sorted(all_datetimes)
        if not all_datetimes:
            logger.error("No data available for backtest.")
            return
        # Initialize equity and open positions
        initial_balance = self.config.get("initial_balance", 10000)
        equity = initial_balance
        open_positions = []  # Each position: dict with symbol, direction, entry_price, size, entry_time, stop_loss, take_profit
        self.equity_curve = []
        self.trade_log = []
        # Main simulation loop
        for i, dt in enumerate(all_datetimes):
            # Build market_data up to current bar for each symbol/timeframe
            market_data = {}
            for symbol in self.symbols:
                market_data[symbol] = {}
                for tf in self.timeframes:
                    df = data[symbol][tf]
                    if not df.empty:
                        # Only include data up to and including current datetime
                        market_data[symbol][tf] = df[df.index <= dt].copy()
            # Generate signals for this bar
            try:
                signals = await self.strategy.generate_signals(market_data=market_data, skip_plots=True, debug_visualize=False, profile=self.config.get('profile', False))
            except Exception as e:
                logger.warning(f"Error generating signals at {dt}: {e}")
                signals = []
            # Simulate fills: market order at next bar open (if possible)
            for signal in signals:
                symbol = signal.get("symbol")
                direction = signal.get("direction")
                entry_price = signal.get("entry_price")
                stop_loss = signal.get("stop_loss")
                take_profit = signal.get("take_profit")
                # Use risk manager for position sizing if available
                if self.risk_manager is not None:
                    size = self.risk_manager.calculate_position_size(
                        account_balance=equity,
                        risk_per_trade=self.config.get("risk_per_trade", 1.0),
                        entry_price=entry_price,
                        stop_loss_price=stop_loss,
                        symbol=symbol
                    )
                else:
                    size = signal.get("size", 1.0)  # Default size 1.0
                tf = signal.get("timeframe", self.timeframes[0])
                df = data[symbol][tf]
                idx = df.index.get_loc(dt) if dt in df.index else None
                if idx is not None and idx + 1 < len(df):
                    next_open = df.iloc[idx + 1]["open"]
                    entry_time = df.index[idx + 1]
                    open_positions.append({
                        "symbol": symbol,
                        "direction": direction,
                        "entry_price": next_open,
                        "size": size,
                        "entry_time": entry_time,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "open": True,
                        "signal_time": dt
                    })
            # Update open positions: check for exits (stop loss, take profit, or end of data)
            for pos in open_positions:
                if not pos["open"]:
                    continue
                symbol = pos["symbol"]
                tf = self.timeframes[0]  # Use primary timeframe for exit
                df = data[symbol][tf]
                if dt not in df.index:
                    continue
                idx = df.index.get_loc(dt)
                if idx < 0:
                    continue
                bar = df.iloc[idx]
                exit_price = None
                exit_reason = None
                if pos["direction"] == "buy":
                    if pos["stop_loss"] is not None and bar["low"] <= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "stop_loss"
                    elif pos["take_profit"] is not None and bar["high"] >= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "take_profit"
                elif pos["direction"] == "sell":
                    if pos["stop_loss"] is not None and bar["high"] >= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "stop_loss"
                    elif pos["take_profit"] is not None and bar["low"] <= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "take_profit"
                if i == len(all_datetimes) - 1 and exit_price is None:
                    exit_price = bar["close"]
                    exit_reason = "end"
                if exit_price is not None:
                    if pos["direction"] == "buy":
                        pnl = (exit_price - pos["entry_price"]) * pos["size"]
                    else:
                        pnl = (pos["entry_price"] - exit_price) * pos["size"]
                    equity += pnl
                    pos["open"] = False
                    pos["exit_price"] = exit_price
                    pos["exit_time"] = dt
                    pos["exit_reason"] = exit_reason
                    pos["pnl"] = pnl
                    self.trade_log.append({
                        "symbol": pos["symbol"],
                        "direction": pos["direction"],
                        "entry_price": pos["entry_price"],
                        "size": pos["size"],
                        "entry_time": pos["entry_time"],
                        "stop_loss": pos["stop_loss"],
                        "take_profit": pos["take_profit"],
                        "exit_price": exit_price,
                        "exit_time": dt,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "signal_time": pos.get("signal_time", None)
                    })
            # Remove closed positions
            open_positions = [p for p in open_positions if p["open"]]
            # Record equity
            self.equity_curve.append({"datetime": dt, "equity": equity})
        # Store results
        self.results = {
            "final_equity": equity,
            "trade_log": self.trade_log,
            "equity_curve": self.equity_curve
        }
        # If performance tracker is provided, update it
        if self.performance_tracker is not None:
            try:
                self.performance = await self.performance_tracker.update_performance_metrics()
            except Exception as e:
                logger.warning(f"Error updating performance tracker: {e}")
        logger.info(f"Backtest complete. Final equity: {equity}")

    def report(self):
        """
        Generate and print/save performance reports and plots.
        Prints summary statistics, outputs trade log, and plots equity curve and drawdown.
        Exports trade log/results as CSV/JSON if configured.
        Computes advanced metrics natively if no PerformanceTracker is provided.
        """
        # Check if results are available
        if not self.results:
            logger.warning("No results to report. Run the backtest first.")
            return
        initial_balance = self.config.get("initial_balance", 10000)
        final_equity = self.results["final_equity"]
        trade_log = pd.DataFrame(self.results["trade_log"])
        equity_curve = pd.DataFrame(self.results["equity_curve"])
        # Defensive check for 'pnl' column
        if "pnl" not in trade_log.columns:
            print("WARNING: No 'pnl' column in trade log. Detailed analytics skipped.")
            print(f"Trade Log (all {len(trade_log)} rows):")
            print(trade_log.to_string(index=False))
            return
        # Summary statistics
        total_trades = len(trade_log)
        win_trades = 0
        loss_trades = 0
        profit_trades = []
        loss_trades_list = []
        win_rate = None
        avg_pnl = None
        profit_factor = None
        expectancy = None
        sharpe_ratio = None
        max_drawdown = None
        returns = []
        # Calculate PnL, win/loss, profit factor, expectancy, Sharpe
        if "pnl" in trade_log.columns and not trade_log.empty:
            win_trades = (trade_log["pnl"] > 0).sum()
            loss_trades = (trade_log["pnl"] < 0).sum()
            profit_trades = trade_log[trade_log["pnl"] > 0]["pnl"].values
            loss_trades_list = trade_log[trade_log["pnl"] < 0]["pnl"].values
            win_rate = win_trades / total_trades if total_trades > 0 else None
            avg_pnl = trade_log["pnl"].mean()
            gross_profit = np.sum(np.array(profit_trades)) if len(profit_trades) > 0 else 0
            gross_loss = -np.sum(np.array(loss_trades_list)) if len(loss_trades_list) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
            expectancy = trade_log["pnl"].mean() if total_trades > 0 else None
        # Calculate equity curve returns and Sharpe ratio
        if not equity_curve.empty:
            equity_curve["cummax"] = equity_curve["equity"].cummax()
            equity_curve["drawdown"] = equity_curve["equity"] - equity_curve["cummax"]
            max_drawdown = equity_curve["drawdown"].min()
            # Calculate returns for Sharpe (simple diff, not log)
            returns = equity_curve["equity"].diff().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 252 trading days
        # Print summary
        print("\n===== Backtest Summary =====")
        print(f"Initial Equity: {initial_balance}")
        print(f"Final Equity:   {final_equity}")
        print(f"Total Trades:   {total_trades}")
        if win_rate is not None:
            print(f"Win Rate:       {win_rate:.2%}")
        if avg_pnl is not None:
            print(f"Average PnL:    {avg_pnl:.2f}")
        if profit_factor is not None:
            print(f"Profit Factor:  {profit_factor:.2f}")
        if expectancy is not None:
            print(f"Expectancy:     {expectancy:.2f}")
        if sharpe_ratio is not None:
            print(f"Sharpe Ratio:   {sharpe_ratio:.2f}")
        if max_drawdown is not None:
            print(f"Max Drawdown:   {max_drawdown:.2f}")
        # If performance tracker is available, print advanced metrics
        if self.performance is not None:
            print("\n--- Advanced Performance Metrics (PerformanceTracker) ---")
            for k, v in self.performance.items():
                print(f"{k}: {v}")
        print("===========================\n")
        # --- DETAILED ANALYTICS ---
        if not trade_log.empty:
            print("Detailed Trade Analytics:")
            # Direction breakdown
            if "direction" in trade_log.columns:
                buy_trades = trade_log[trade_log["direction"] == "buy"]
                sell_trades = trade_log[trade_log["direction"] == "sell"]
                print(f"  Buy trades: {len(buy_trades)} (Win rate: {((buy_trades['pnl'] > 0).mean() * 100 if not buy_trades.empty else 0):.2f}%)")
                print(f"  Sell trades: {len(sell_trades)} (Win rate: {((sell_trades['pnl'] > 0).mean() * 100 if not sell_trades.empty else 0):.2f}%)")
            # Largest win/loss
            print(f"  Largest Win: {trade_log['pnl'].max():.2f}")
            print(f"  Largest Loss: {trade_log['pnl'].min():.2f}")
            # Average holding time
            if "entry_time" in trade_log.columns and "exit_time" in trade_log.columns:
                holding_times = pd.to_datetime(trade_log["exit_time"]) - pd.to_datetime(trade_log["entry_time"])
                avg_holding = holding_times.mean()
                print(f"  Average Holding Time: {avg_holding}")
            # Distribution of returns
            print(f"  Median PnL: {trade_log['pnl'].median():.2f}")
            print(f"  25th/75th Percentile PnL: {trade_log['pnl'].quantile(0.25):.2f} / {trade_log['pnl'].quantile(0.75):.2f}")
            # Streaks
            pnl_sign = (trade_log['pnl'] > 0).astype(int)
            streaks = (pnl_sign != pnl_sign.shift()).cumsum()
            win_streaks = pnl_sign.groupby(streaks).sum() * pnl_sign.groupby(streaks).size()
            lose_streaks = (1-pnl_sign).groupby(streaks).sum() * (1-pnl_sign).groupby(streaks).size()
            print(f"  Max Winning Streak: {win_streaks.max() if not win_streaks.empty else 0}")
            print(f"  Max Losing Streak: {lose_streaks.max() if not lose_streaks.empty else 0}")
            # Risk/Reward analytics
            if "stop_loss" in trade_log.columns and "take_profit" in trade_log.columns and "entry_price" in trade_log.columns:
                rr_ratios = np.abs((trade_log["take_profit"] - trade_log["entry_price"]) / (trade_log["entry_price"] - trade_log["stop_loss"]))
                print(f"  Avg Risk/Reward Ratio: {rr_ratios.mean():.2f}")
            # TP/SL hit rates
            if "exit_reason" in trade_log.columns:
                tp_hits = (trade_log["exit_reason"] == "take_profit").sum()
                sl_hits = (trade_log["exit_reason"] == "stop_loss").sum()
                print(f"  TP Hit Rate: {tp_hits / total_trades:.2%}")
                print(f"  SL Hit Rate: {sl_hits / total_trades:.2%}")
            # Expectancy per trade
            if win_rate is not None and avg_pnl is not None:
                avg_win = trade_log[trade_log["pnl"] > 0]["pnl"].mean() if win_trades > 0 else 0
                avg_loss = trade_log[trade_log["pnl"] < 0]["pnl"].mean() if loss_trades > 0 else 0
                expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
                print(f"  Expectancy per Trade: {expectancy:.2f}")
            # Equity/Drawdown analytics
            if not equity_curve.empty:
                max_equity = equity_curve["equity"].max()
                min_equity = equity_curve["equity"].min()
                max_equity_time = equity_curve.loc[equity_curve["equity"].idxmax(), "datetime"]
                min_equity_time = equity_curve.loc[equity_curve["equity"].idxmin(), "datetime"]
                print(f"  Max Equity: {max_equity:.2f} at {max_equity_time}")
                print(f"  Min Equity: {min_equity:.2f} at {min_equity_time}")
                # Drawdown duration
                drawdown_periods = equity_curve["drawdown"] < 0
                if drawdown_periods.any():
                    dd_starts = equity_curve["drawdown"][drawdown_periods].index.to_list()
                    print(f"  Drawdown Periods: {len(dd_starts)}")
                # Recovery factor
                net_profit = final_equity - initial_balance
                recovery_factor = net_profit / abs(max_drawdown) if max_drawdown else np.nan
                print(f"  Recovery Factor: {recovery_factor:.2f}")
            # Time-based analytics
            if "exit_time" in trade_log.columns and "pnl" in trade_log.columns:
                trade_log["exit_time"] = pd.to_datetime(trade_log["exit_time"])
                trade_log["month"] = trade_log["exit_time"].dt.to_period("M")
                trade_log["day"] = trade_log["exit_time"].dt.date
                monthly_returns = trade_log.groupby("month")["pnl"].sum()
                daily_returns = trade_log.groupby("day")["pnl"].sum()
                print(f"  Best Month: {monthly_returns.idxmax()} ({monthly_returns.max():.2f})")
                print(f"  Worst Month: {monthly_returns.idxmin()} ({monthly_returns.min():.2f})")
                print(f"  Best Day: {daily_returns.idxmax()} ({daily_returns.max():.2f})")
                print(f"  Worst Day: {daily_returns.idxmin()} ({daily_returns.min():.2f})")
            print("---------------------------\n")
            # --- Visualizations ---
            # Histogram of trade PnL
            plt.figure(figsize=(8, 4))
            plt.hist(trade_log["pnl"], bins=30, color="skyblue", edgecolor="black")
            plt.title("Histogram of Trade PnL")
            plt.xlabel("PnL")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
            # Cumulative PnL by trade
            plt.figure(figsize=(10, 4))
            plt.plot(trade_log["pnl"].cumsum(), label="Cumulative PnL")
            plt.title("Cumulative PnL by Trade")
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative PnL")
            plt.legend()
            plt.grid(True)
            plt.show()
            # Monthly returns bar chart
            if "month" in trade_log.columns:
                monthly_returns = trade_log.groupby("month")["pnl"].sum()
                plt.figure(figsize=(10, 4))
                monthly_returns.plot(kind="bar", color="orange")
                plt.title("Monthly Returns")
                plt.xlabel("Month")
                plt.ylabel("Total PnL")
                plt.grid(True)
                plt.show()
        # Output trade log
        print(f"Trade Log (all {len(trade_log)} rows):")
        print(trade_log.to_string(index=False))
        # Save trade log as CSV if path provided
        trade_log_path = self.config.get("trade_log_path")
        if trade_log_path:
            trade_log.to_csv(trade_log_path, index=False)
            print(f"Trade log saved to {trade_log_path}")
        # Save trade log as JSON if path provided
        trade_log_json_path = self.config.get("trade_log_json_path")
        if trade_log_json_path:
            # Drop or convert non-serializable columns before saving
            json_safe_trade_log = trade_log.copy()
            if "month" in json_safe_trade_log.columns:
                json_safe_trade_log = json_safe_trade_log.drop(columns=["month"])
            if "day" in json_safe_trade_log.columns:
                json_safe_trade_log["day"] = json_safe_trade_log["day"].astype(str)
            json_safe_trade_log.to_json(trade_log_json_path, orient="records", date_format="iso")
            print(f"Trade log saved to {trade_log_json_path}")
        # Save results as JSON if path provided
        results_json_path = self.config.get("results_json_path")
        if results_json_path:
            with open(results_json_path, "w") as f:
                json.dump(self.results, f, default=str, indent=2)
            print(f"Results saved to {results_json_path}")
        # Plot equity curve and drawdown
        if not equity_curve.empty:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(equity_curve["datetime"], equity_curve["equity"], label="Equity Curve", color="blue")
            ax1.set_xlabel("Datetime")
            ax1.set_ylabel("Equity", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax2 = ax1.twinx()
            ax2.plot(equity_curve["datetime"], equity_curve["drawdown"], label="Drawdown", color="red", alpha=0.5)
            ax2.set_ylabel("Drawdown", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            plt.title("Equity Curve and Drawdown")
            fig.tight_layout()
            plt.grid(True)
            plt.show()
        if "symbol" in trade_log.columns and "pnl" in trade_log.columns:
            print("\nPair Performance Breakdown:")
            pair_stats = trade_log.groupby("symbol").agg(
                total_trades=("pnl", "count"),
                win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                max_win=("pnl", "max"),
                max_loss=("pnl", "min")
            ).sort_values("total_pnl", ascending=False)
            print(pair_stats.to_string(float_format=lambda x: f"{x:.2f}"))
            print(f"\nBest Performing Pair: {pair_stats.index[0]} (Total PnL: {pair_stats.iloc[0]['total_pnl']:.2f})")
            print(f"Worst Performing Pair: {pair_stats.index[-1]} (Total PnL: {pair_stats.iloc[-1]['total_pnl']:.2f})")

    @staticmethod
    def load_csv_data(directory: str, symbols: List[str], timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load OHLCV data from CSV files for all symbols and timeframes.
        Args:
            directory: Path to the folder containing CSV files
            symbols: List of symbols
            timeframes: List of timeframes
        Returns:
            Nested dict: {symbol: {timeframe: DataFrame}}
        """
        data = {}
        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                filename = os.path.join(directory, f"{symbol}_{tf}.csv")
                try:
                    df = pd.read_csv(filename, parse_dates=["datetime"])
                    df.set_index("datetime", inplace=True)
                    data[symbol][tf] = df
                    logger.info(f"Loaded {len(df)} rows for {symbol} {tf} from {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    data[symbol][tf] = pd.DataFrame()
        return data

    @staticmethod
    def load_mt5_data(mt5_handler, symbols: List[str], timeframes: List[str], start_date=None, end_date=None, num_bars: int = 1000) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load OHLCV data from MT5 for all symbols and timeframes, by date range if provided.
        """
        data = {}
        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                try:
                    if start_date and end_date:
                        df = mt5_handler.get_historical_data(symbol, tf, start_date, end_date)
                    else:
                        df = mt5_handler.get_market_data(symbol, tf, num_bars)
                    if df is not None and not df.empty:
                        data[symbol][tf] = df
                        logger.info(f"Loaded {len(df)} bars for {symbol} {tf} from MT5")
                    else:
                        logger.warning(f"No data returned for {symbol} {tf} from MT5")
                        data[symbol][tf] = pd.DataFrame()
                except Exception as e:
                    logger.warning(f"Failed to load {symbol} {tf} from MT5: {e}")
                    data[symbol][tf] = pd.DataFrame()
        return data

    @staticmethod
    def load_or_fetch_ohlcv_data(directory: str, mt5_handler, symbols: list, timeframes: list, start_date=None, end_date=None, num_bars: int = 1000) -> dict:
        """
        Load OHLCV data from CSV if available, otherwise fetch from MT5, save as CSV, and use it. Use date range if provided.
        NOW: If a cached file has no rows within the requested date range, or sanitisation removes all rows (e.g. all-zero OHLC), we automatically re-fetch from MT5. A light sanitiser removes rows with NaN or 0 for any of the OHLC columns.
        """
        import os
        import pandas as pd
        data = {}
        os.makedirs(directory, exist_ok=True)
        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                filename = os.path.join(directory, f"{symbol}_{tf}.csv")
                needs_refetch = False
                df = None
                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename, parse_dates=["datetime"])
                        if "datetime" not in df.columns:
                            print(f"WARNING: {filename} missing 'datetime' column. Will re-fetch from MT5.")
                            needs_refetch = True
                        elif df.empty:
                            print(f"WARNING: {filename} is empty. Will re-fetch from MT5.")
                            needs_refetch = True
                        else:
                            df.set_index("datetime", inplace=True)
                            if start_date and end_date:
                                # Filter to requested date range *before* accepting cache
                                df = df[(df.index >= start_date) & (df.index <= end_date)]
                                if df.empty:
                                    logger.warning(f"Cached file {filename} has no data in requested range. Will re-fetch from MT5.")
                                    needs_refetch = True
                                    df = None
                            if not needs_refetch and df is not None:
                                # --- Basic sanitisation: drop rows where any OHLC column is 0/NaN ---
                                required_cols = ['open', 'high', 'low', 'close']
                                if all(col in df.columns for col in required_cols):
                                    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
                                    before_rows = len(df)
                                    df = df.dropna(subset=required_cols)
                                    df = df[(df[required_cols] != 0).all(axis=1)]
                                    after_rows = len(df)
                                    if after_rows == 0:
                                        logger.warning(f"Cached file {filename} contained only zero/NaN rows after sanitisation – will re-fetch.")
                                        needs_refetch = True
                                        df = None
                                    elif after_rows < before_rows:
                                        logger.info(f"Sanitised {filename}: removed {before_rows - after_rows} bad rows (now {after_rows}).")
                                else:
                                    logger.warning(f"Cached file {filename} missing OHLC columns – will re-fetch.")
                                    needs_refetch = True
                                    df = None
                                if not needs_refetch and df is not None:
                                    data[symbol][tf] = df
                                    logger.info(f"Loaded {len(df)} rows for {symbol} {tf} from {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}. Will re-fetch from MT5.")
                        needs_refetch = True
                else:
                    needs_refetch = True
                if needs_refetch:
                    # Lazy MT5Handler creation
                    if mt5_handler is None:
                        mt5_handler = MT5Handler.get_instance()
                    try:
                        if start_date and end_date:
                            df = mt5_handler.get_historical_data(symbol, tf, start_date, end_date)
                        else:
                            df = mt5_handler.get_market_data(symbol, tf, num_bars)
                        if df is not None and not df.empty:
                            # Ensure 'datetime' is a column for saving
                            if 'datetime' not in df.columns:
                                if df.index.name == "datetime":
                                    df = df.reset_index()
                                elif 'time' in df.columns:
                                    df = df.rename(columns={'time': 'datetime'})
                                elif df.index.name == "time":
                                    df = df.reset_index().rename(columns={'time': 'datetime'})
                                else:
                                    # fallback: try to create a datetime column from index
                                    df['datetime'] = df.index
                            df.to_csv(filename, index=False)
                            df.set_index("datetime", inplace=True)
                            # Final sanitisation identical to cache path
                            required_cols = ['open', 'high', 'low', 'close']
                            if all(col in df.columns for col in required_cols):
                                df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
                                before_rows = len(df)
                                df = df.dropna(subset=required_cols)
                                df = df[(df[required_cols] != 0).all(axis=1)]
                                if len(df) < before_rows:
                                    logger.info(f"Sanitised freshly fetched {symbol} {tf}: removed {before_rows - len(df)} bad rows (now {len(df)}).")
                            data[symbol][tf] = df
                            logger.info(f"Fetched and saved {len(df)} bars for {symbol} {tf} to {filename}")
                        else:
                            logger.warning(f"No data returned for {symbol} {tf} from MT5")
                            data[symbol][tf] = pd.DataFrame()
                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol} {tf} from MT5: {e}")
                        data[symbol][tf] = pd.DataFrame()
        return data

    # TODO: Add more helper methods as needed (trade simulation, position management, etc.)

def load_strategy(strategy_name: str, params: dict = {}):
    """
    Dynamically load and instantiate a strategy class from src/strategy/ by name.
    Args:
        strategy_name: Name of the strategy class (e.g., 'BreakoutReversalStrategy')
        params: Optional dict of parameters to pass to the constructor
    Returns:
        An instance of the strategy class
    Raises:
        ImportError if the strategy class is not found
    """
    import importlib
    import sys
    strategy_module_path = f"src.strategy.{strategy_name.lower()}"
    # Try to import the module by file name (kebab/snake case)
    try:
        # Try snake_case first
        module_name = f"src.strategy.{strategy_name.lower()}"
        if module_name not in sys.modules:
            strategy_module = importlib.import_module(module_name)
        else:
            strategy_module = sys.modules[module_name]
    except ImportError:
        # Try direct import from strategy/__init__.py
        try:
            strategy_module = importlib.import_module("src.strategy")
        except ImportError:
            raise ImportError(f"Could not import strategy module for {strategy_name}")
    # Try to get the class from the module
    try:
        strategy_class = getattr(strategy_module, strategy_name)
    except AttributeError:
        raise ImportError(f"Strategy class '{strategy_name}' not found in module {strategy_module}")
    # Instantiate with params
    return strategy_class(**params)

def select_data_loader(mode: str, **kwargs):
    start_date = kwargs.get('start_date')
    end_date = kwargs.get('end_date')
    if mode == 'csv':
        directory = kwargs.get('directory')
        if not directory:
            raise ValueError("CSV data loader requires 'directory' argument.")
        return lambda symbols, timeframes: Backtester.load_csv_data(directory, symbols, timeframes)
    elif mode == 'mt5':
        mt5_handler = kwargs.get('mt5_handler')
        num_bars = kwargs.get('num_bars', 1000)
        if not mt5_handler:
            raise ValueError("MT5 data loader requires 'mt5_handler' argument.")
        return lambda symbols, timeframes: Backtester.load_mt5_data(mt5_handler, symbols, timeframes, start_date=start_date, end_date=end_date, num_bars=num_bars)
    elif mode == 'cache':
        directory = kwargs.get('directory')
        mt5_handler = kwargs.get('mt5_handler')
        num_bars = kwargs.get('num_bars', 1000)
        if not directory:
            raise ValueError("Cache data loader requires 'directory' argument.")
        return lambda symbols, timeframes: Backtester.load_or_fetch_ohlcv_data(directory, mt5_handler, symbols, timeframes, start_date=start_date, end_date=end_date, num_bars=num_bars)
    else:
        raise ValueError(f"Unknown data loader mode: {mode}")

async def batch_backtest(configs: list):
    """
    Run multiple backtests in sequence and collect results.
    Args:
        configs: List of dicts, each with keys:
            - strategy_name: str
            - strategy_params: dict
            - data_loader: function
            - symbols: list
            - timeframes: list
            - config: dict
            - risk_manager: optional
            - performance_tracker: optional
    Returns:
        List of results (dicts) for each run
    """
    import pandas as pd
    results = []
    summary_rows = []
    for i, cfg in enumerate(configs):
        print(f"\n=== Running Backtest {i+1}/{len(configs)}: {cfg['strategy_name']} ===")
        # Load strategy
        strategy = load_strategy(cfg['strategy_name'], cfg.get('strategy_params', {}))
        # Create backtester
        backtester = Backtester(
            strategy=strategy,
            data_loader=cfg['data_loader'],
            symbols=cfg['symbols'],
            timeframes=cfg['timeframes'],
            config=cfg.get('config', {}),
            risk_manager=cfg.get('risk_manager'),
            performance_tracker=cfg.get('performance_tracker')
        )
        # Run backtest
        await backtester.run()
        # Collect results
        res = {
            'strategy': cfg['strategy_name'],
            'params': cfg.get('strategy_params', {}),
            'final_equity': backtester.results['final_equity'] if backtester.results else None,
            'total_trades': len(backtester.results['trade_log']) if backtester.results else 0,
            'win_rate': None,
            'max_drawdown': None
        }
        # Calculate win rate and max drawdown if possible
        trade_log = pd.DataFrame(backtester.results['trade_log']) if backtester.results else pd.DataFrame()
        if 'pnl' in trade_log.columns and not trade_log.empty:
            win_trades = (trade_log['pnl'] > 0).sum()
            res['win_rate'] = win_trades / len(trade_log) if len(trade_log) > 0 else None
        equity_curve = pd.DataFrame(backtester.results['equity_curve']) if backtester.results else pd.DataFrame()
        if not equity_curve.empty:
            equity_curve['cummax'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = equity_curve['equity'] - equity_curve['cummax']
            res['max_drawdown'] = equity_curve['drawdown'].min()
        results.append(res)
        summary_rows.append([
            cfg['strategy_name'],
            str(cfg.get('strategy_params', {})),
            res['final_equity'],
            res['total_trades'],
            f"{res['win_rate']:.2%}" if res['win_rate'] is not None else '-',
            f"{res['max_drawdown']:.2f}" if res['max_drawdown'] is not None else '-'
        ])
    # Print summary table
    print("\n=== Batch Backtest Summary ===")
    summary_df = pd.DataFrame(summary_rows, columns=[
        'Strategy', 'Params', 'Final Equity', 'Total Trades', 'Win Rate', 'Max Drawdown'
    ])
    print(summary_df)
    return results 