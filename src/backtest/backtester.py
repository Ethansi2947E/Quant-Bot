"""
Backtester module for trading strategies.

# TODO: For future: support event-driven batching, parallelized signal generation, and database export.

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
import backtrader as bt
import asyncio # For running async strategy methods
from src.mt5_handler import MT5Handler

class BacktraderStrategyAdapter(bt.Strategy):
    params = (
        ('original_strategy', None),
        ('data_feed_map', None), # Maps data feed index to (symbol, tf, original_df_ref)
        ('strategy_kwargs', None), # kwargs for generate_signals
    )

    def __init__(self):
        self.original_strategy = self.p.original_strategy
        self.trade_history = []
        # Create a new event loop for this strategy instance if original_strategy.generate_signals is async
        if asyncio.iscoroutinefunction(self.original_strategy.generate_signals):
            self.loop = asyncio.new_event_loop()
        else:
            self.loop = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else: # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.Status[order.status]}')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        self.trade_history.append({
            'symbol': trade.data._name.split('_')[0], # Assuming name is "SYMBOL_TF"
            'direction': 'buy' if trade.history[0].event.order.isbuy() else 'sell',
            'entry_price': trade.price,
            'exit_price': trade.history[-1].event.price, # Last event price should be exit
            'size': trade.size,
            'pnl': trade.pnl,
            'pnl_comm': trade.pnlcomm,
            'open_dt': bt.num2date(trade.history[0].status.dt).isoformat(),
            'close_dt': bt.num2date(trade.history[-1].status.dt).isoformat(),
        })

    def next(self):
        market_data = {}
        current_max_dt = None

        # Determine the current maximum datetime across all feeds for this step
        for data_feed in self.datas:
            try:
                # Check if data_feed.datetime is valid and has data
                if len(data_feed.datetime) > 0 :
                    dt_obj = data_feed.datetime.datetime(0)
                    if current_max_dt is None or dt_obj > current_max_dt:
                        current_max_dt = dt_obj
                else: # No data yet for this feed
                    return 
            except IndexError: # datetime buffer may not be populated yet
                return

        if current_max_dt is None: # Should not happen if above checks pass
            return

        # Construct market_data for the original strategy
        for i, data_feed in enumerate(self.datas):
            symbol, tf, original_df_ref = self.p.data_feed_map[i]
            
            # Slice the original DataFrame up to the current_max_dt
            # This ensures all dataframes in market_data are aligned to the same "current time"
            df_slice = original_df_ref[original_df_ref.index <= current_max_dt].copy()

            if symbol not in market_data:
                market_data[symbol] = {}
            market_data[symbol][tf] = df_slice

        # Call the original strategy's signal generation
        strategy_kwargs = self.p.strategy_kwargs or {}
        if self.loop: # If original strategy is async
            signals = self.loop.run_until_complete(
                self.original_strategy.generate_signals(market_data=market_data, **strategy_kwargs)
            )
        else: # If original strategy is sync
            signals = self.original_strategy.generate_signals(market_data=market_data, **strategy_kwargs)

        # Process signals
        for signal in signals:
            symbol_signal = signal.get("symbol")
            direction = signal.get("direction")
            entry_price = signal.get("entry_price") # May not be used by market order
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")
            size = signal.get("size", 1.0) # Default size if not specified

            # Find the correct data feed for the signal's symbol (assuming primary timeframe for orders)
            target_data_feed = None
            for i, data_feed_obj in enumerate(self.datas):
                s, _, _ = self.p.data_feed_map[i]
                # TODO: This assumes orders are placed on the first data feed for that symbol.
                # More robust mapping might be needed if strategies generate signals for specific timeframes.
                if s == symbol_signal:
                    target_data_feed = data_feed_obj
                    break
            
            if target_data_feed is None:
                self.log(f"Could not find data feed for symbol {symbol_signal} in signal. Skipping.")
                continue

            if direction == "buy":
                self.buy(data=target_data_feed, size=size, exectype=bt.Order.Market, transmit=True, sl=stop_loss, tp=take_profit)
                self.log(f"BUY ORDER: {symbol_signal}, Size: {size}, SL: {stop_loss}, TP: {take_profit}")
            elif direction == "sell":
                self.sell(data=target_data_feed, size=size, exectype=bt.Order.Market, transmit=True, sl=stop_loss, tp=take_profit)
                self.log(f"SELL ORDER: {symbol_signal}, Size: {size}, SL: {stop_loss}, TP: {take_profit}")

    def stop(self):
        if self.loop:
            self.loop.close()
        self.log('Strategy STOP called. Final Portfolio Value: %.2f' % self.broker.getvalue())

class Backtester:
    """
    Core backtesting engine for trading strategies.
    
    Usage:
        - Initialize with strategy, data loader, config, and symbols
        - Call run() to execute the backtest
        - Access results and reports after completion
    
    Note: If 'profile': True is set in config, profiling will be enabled for strategy.generate_signals,
    and parallelization will be automatically disabled to avoid cProfile conflicts.
    
    # NOTE: If using BreakoutTradingStrategy, default settings now use a looser consolidation filter (bb_squeeze_factor=1.2, min_consolidation_bars=5),
    # a more permissive volume filter (>=, 0.9x tolerance), and debugging aids (processed_bars clearing, wait_for_confirmation_candle=False).
    """
    def __init__(self, 
                 strategy, 
                 data_loader: Callable, 
                 symbols: List[str],
                 timeframes: List[str],
                 config: Optional[Dict] = None,
                 risk_manager: Optional[Any] = None,
                 performance_tracker: Optional[Any] = None,
                 export_config: Optional[Dict] = None):
        """
        Args:
            strategy: An instance of a SignalGenerator-compatible strategy
            data_loader: Function to load historical data (CSV or MT5)
            symbols: List of symbols to backtest
            timeframes: List of timeframes required by the strategy
            config: Optional configuration dictionary
            risk_manager: Optional RiskManager instance for position sizing
            performance_tracker: Optional PerformanceTracker instance for advanced metrics
            export_config: Optional dict for export settings (paths, formats)
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
        self.cerebro = None # Will hold the backtrader Cerebro instance
        self.run_results = None # To store results from cerebro.run()
        # Unified export config
        self.export_config = export_config or self.config.get('export_config', {})
        # TODO: Add more attributes as needed (risk manager, performance tracker, etc.)

    def run(self):
        """
        Main backtesting loop. Simulates bar-by-bar trading for all symbols.
        Loads data, aligns by datetime, iterates through each bar, and simulates trades.
        Stores results in self.results, self.trade_log, and self.equity_curve.
        
        Note: If profiling is enabled, parallelization is automatically disabled by the strategy.
        """
        self.cerebro = bt.Cerebro()

        # Load data using the provided data_loader
        loaded_data = self.data_loader(self.symbols, self.timeframes)
        
        data_feed_map = {} # Maps cerebro data index to (symbol, tf, original_df_ref)
        data_idx_counter = 0

        for symbol_name in self.symbols:
            if symbol_name not in loaded_data:
                logger.warning(f"No data loaded for symbol: {symbol_name}")
                continue
            for tf_name in self.timeframes: # Assuming self.timeframes contains all TFs needed
                if tf_name not in loaded_data[symbol_name]:
                    logger.warning(f"No data loaded for {symbol_name} timeframe: {tf_name}")
                    continue
                
                df = loaded_data[symbol_name][tf_name]
                if df is not None and not df.empty:
                    # Ensure df.index is datetime
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    # Ensure standard column names for backtrader
                    # open, high, low, close, volume, openinterest
                    required_cols = {'open', 'high', 'low', 'close', 'volume'}
                    if not required_cols.issubset(df.columns):
                        logger.error(f"DataFrame for {symbol_name}/{tf_name} is missing required columns. Has: {df.columns}. Needs: {required_cols}")
                    continue
                        
                    if 'openinterest' not in df.columns:
                        df['openinterest'] = 0
                    
                    # bt.feeds.PandasData expects datetime index to be named 'datetime' if not already
                    df.index.name = 'datetime'

                    data_feed = bt.feeds.PandasData(dataname=df) # type: ignore
                    self.cerebro.adddata(data_feed, name=f"{symbol_name}_{tf_name}")
                    data_feed_map[data_idx_counter] = (symbol_name, tf_name, df) # Store original df for adapter
                    data_idx_counter += 1
                else:
                    logger.warning(f"Empty DataFrame for {symbol_name}/{tf_name}")

        if not data_feed_map:
            logger.error("No data feeds added to Cerebro. Aborting backtest.")
            return

        # Add strategy adapter
        strategy_kwargs_for_adapter = {
            'skip_plots': True, # Example: pass relevant kwargs
            'debug_visualize': False,
            'profile': self.config.get('profile', False)
        }
        self.cerebro.addstrategy(BacktraderStrategyAdapter, 
                                 original_strategy=self.strategy, 
                                 data_feed_map=data_feed_map,
                                 strategy_kwargs=strategy_kwargs_for_adapter)

        initial_balance = self.config.get("initial_balance", 10000)
        self.cerebro.broker.setcash(initial_balance)
        # TODO: Add commission, slippage if needed
        # self.cerebro.broker.setcommission(commission=0.001) 

        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days) # type: ignore # Adjust timeframe as needed
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

        logger.info(f"Starting backtrader Cerebro engine with initial balance: {initial_balance}")
        self.run_results = self.cerebro.run()
        logger.info(f"Backtrader Cerebro engine finished. Final portfolio value: {self.cerebro.broker.getvalue():.2f}")

        # Store results from analyzers
        if self.run_results and self.run_results[0]:
            strategy_instance = self.run_results[0]
            self.results = {
                "final_equity": self.cerebro.broker.getvalue(),
                "trade_analyzer": strategy_instance.analyzers.tradeanalyzer.get_analysis() if hasattr(strategy_instance.analyzers, 'tradeanalyzer') else None,
                "sharpe_ratio": strategy_instance.analyzers.sharpe.get_analysis() if hasattr(strategy_instance.analyzers, 'sharpe') else None,
                "drawdown": strategy_instance.analyzers.drawdown.get_analysis() if hasattr(strategy_instance.analyzers, 'drawdown') else None,
                "sqn": strategy_instance.analyzers.sqn.get_analysis() if hasattr(strategy_instance.analyzers, 'sqn') else None,
                "trade_log": strategy_instance.trade_history if hasattr(strategy_instance, 'trade_history') else []
            }
            self.trade_log = self.results["trade_log"]

    def report(self):
        """
        Generate and print/save performance reports and plots.
        Prints summary statistics, outputs trade log, and plots equity curve and drawdown.
        Exports trade log/results as CSV/JSON/Excel/Markdown/HTML if configured.
        Computes advanced metrics natively if no PerformanceTracker is provided.
        """
        if not self.cerebro or not self.run_results or not self.results:
            logger.warning("No results to report. Run the backtest first.")
            return

        initial_balance = self.config.get("initial_balance", 10000)
        final_equity = self.results["final_equity"]
        
        trade_analysis = self.results.get("trade_analyzer", {})
        sharpe_analysis = self.results.get("sharpe_ratio", {})
        drawdown_analysis = self.results.get("drawdown", {})
        sqn_analysis = self.results.get("sqn", {})
        
        # Use trade_log from strategy instance if available
        trade_log_list = self.results.get("trade_log", [])
        trade_log_df = pd.DataFrame(trade_log_list)

        # Summary statistics
        total_trades = len(trade_log_df)
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

        if trade_analysis and 'total' in trade_analysis and trade_analysis['total']['total'] > 0:
            total_trades = trade_analysis['total']['total']
            win_trades = trade_analysis['won']['total']
            loss_trades = trade_analysis['lost']['total']
            win_rate = trade_analysis['won']['total'] / trade_analysis['total']['total'] if trade_analysis['total']['total'] > 0 else 0
            avg_pnl = trade_analysis['pnl']['net']['average']
            
            gross_profit = trade_analysis['pnl']['gross']['total']
            gross_loss = abs(trade_analysis['pnl']['gross']['total'] - trade_analysis['pnl']['net']['total']) # Approximate
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            expectancy = avg_pnl # Simplified

        if drawdown_analysis and 'max' in drawdown_analysis:
            max_drawdown = drawdown_analysis['max']['drawdown']
        if sharpe_analysis and 'sharperatio' in sharpe_analysis:
            sharpe_ratio = sharpe_analysis['sharperatio']

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
        if sqn_analysis and 'sqn' in sqn_analysis:
            print(f"SQN:            {sqn_analysis['sqn']:.2f}")
        
        print("===========================\n")

        # --- DETAILED ANALYTICS (from trade_log_df if available) ---
        if not trade_log_df.empty and 'pnl' in trade_log_df.columns:
            trade_log = trade_log_df # Use the DataFrame for easier analysis
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
                print(f"  Avg Risk/Reward Ratio (from signals): {rr_ratios.mean():.2f}") # This is based on signal's SL/TP, not actual fills
            # TP/SL hit rates
            if 'exit_reason' in trade_log.columns: # This column might not exist with backtrader's log
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
        else:
            logger.info("Trade log is empty or missing 'pnl' column. Skipping detailed analytics.")

        # Output trade log
        print(f"Trade Log (all {len(trade_log_df)} rows):")
        print(trade_log_df.to_string(index=False))

        # --- Unified Export Logic ---
        export_cfg = self.export_config
        trade_log_to_export = trade_log_df # Use the DataFrame from strategy's trade_history

        # CSV Export
        csv_path = export_cfg.get('csv') or self.config.get('trade_log_path')
        if csv_path:
            trade_log_to_export.to_csv(csv_path, index=False)
            print(f"Trade log saved to {csv_path}")
        # JSON Export
        json_path = export_cfg.get('json') or self.config.get('trade_log_json_path')
        if json_path:
            json_safe_trade_log = trade_log_to_export.copy()
            if "month" in json_safe_trade_log.columns:
                json_safe_trade_log = json_safe_trade_log.drop(columns=["month"])
            if "day" in json_safe_trade_log.columns:
                json_safe_trade_log["day"] = json_safe_trade_log["day"].astype(str)
            json_safe_trade_log.to_json(json_path, orient="records", date_format="iso")
            print(f"Trade log saved to {json_path}")
        # Excel Export
        excel_path = export_cfg.get('excel')
        if excel_path:
            trade_log_to_export.to_excel(excel_path, index=False)
            print(f"Trade log saved to {excel_path}")
        # Unified machine-readable JSON report
        report_json_path = export_cfg.get('report_json')
        if report_json_path:
            report_obj = {
                'metadata': {
                    'symbols': self.symbols,
                    'timeframes': self.timeframes,
                    'initial_balance': initial_balance,
                    'final_equity': final_equity,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'config': self.config,
                },
                'trade_log': trade_log_to_export.to_dict(orient='records'),
                # 'equity_curve': equity_curve.to_dict(orient='records'), # Equity curve from cerebro plot
            }
            with open(report_json_path, "w") as f:
                json.dump(report_obj, f, default=str, indent=2)
            print(f"Unified report saved to {report_json_path}")
        # Markdown Export
        md_path = export_cfg.get('markdown')
        if md_path:
            with open(md_path, "w") as f:
                f.write(f"# Backtest Summary\n\n")
                f.write(f"**Initial Equity:** {initial_balance}\n\n")
                f.write(f"**Final Equity:** {final_equity}\n\n")
                f.write(f"**Total Trades:** {total_trades}\n\n")
                if win_rate is not None:
                    f.write(f"**Win Rate:** {win_rate:.2%}\n\n")
                if profit_factor is not None:
                    f.write(f"**Profit Factor:** {profit_factor:.2f}\n\n")
                if max_drawdown is not None:
                    f.write(f"**Max Drawdown:** {max_drawdown:.2f}\n\n")
                if sharpe_ratio is not None:
                    f.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}\n\n")
                f.write(f"\n## Trade Log\n\n")
                f.write(trade_log_to_export.to_markdown(index=False))
            print(f"Markdown report saved to {md_path}")
        # HTML Export (stub)
        html_path = export_cfg.get('html')
        if html_path:
            with open(html_path, "w") as f:
                f.write("<h1>Backtest Report</h1>")
                f.write(f"<p>Initial Equity: {initial_balance}</p>")
                f.write(f"<p>Final Equity: {final_equity}</p>")
                # Add more stats if available
                f.write("<h2>Trade Log</h2>")
                f.write(trade_log_to_export.to_html(index=False))
                # Placeholder for plot image if saved
                f.write("<h2>Equity Curve & Drawdown</h2><p>(Plot saved separately if configured)</p>")
            print(f"HTML report stub saved to {html_path}")

        # Use cerebro.plot()
        plot_path = self.export_config.get('plot_path', None) # Example: 'src/backtest/results/.../plot.png'
        self.cerebro.plot(style='candlestick', barup='green', bardown='red', savefig=bool(plot_path), figfilename=plot_path)
        if plot_path: logger.info(f"Backtrader plot saved to {plot_path}")

        if not trade_log_df.empty and "symbol" in trade_log_df.columns and "pnl" in trade_log_df.columns:
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
    def load_or_fetch_ohlcv_data(directory: str, 
                                 mt5_handler_instance: Optional[MT5Handler], 
                                 symbols: list, 
                                 timeframes: list, 
                                 start_date=None, 
                                 end_date=None, 
                                 num_bars: int = 1000,
                                 allow_external_fetch: bool = True) -> dict:
        """
        Load OHLCV data from CSV if available, otherwise fetch from MT5 (if allowed), save as CSV, and use it.
        Args:
            directory: Path to the folder containing CSV files (structured by year/symbol/timeframe)
            mt5_handler_instance: An optional, pre-initialized MT5Handler instance.
            symbols: List of symbols
            timeframes: List of timeframes
            start_date: Optional start date for fetching/filtering data.
            end_date: Optional end date for fetching/filtering data.
            num_bars: Number of bars to fetch if fetching by count (not date range).
            allow_external_fetch: If False, will not attempt to fetch from MT5 if local data is missing.
        Returns:
            Nested dict: {symbol: {timeframe: DataFrame}}
        """
        import os
        # Pandas is already imported at the module level
        # import pandas as pd 
        data = {}
        # Determine year folder from start_date
        year_str = str(start_date.year) if start_date else 'unknown_year' # Ensure year_str is robust

        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                # Build path: {directory}/{year}/{symbol}/{timeframe}.csv
                symbol_dir = os.path.join(directory, year_str, symbol)
                os.makedirs(symbol_dir, exist_ok=True)
                filename = os.path.join(symbol_dir, f"{tf}.csv")
                
                needs_refetch = False
                df = None

                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename, parse_dates=["datetime"])
                        if "datetime" not in df.columns:
                            logger.warning(f"{filename} missing 'datetime' column. Marking for re-fetch.")
                            needs_refetch = True
                            df = None # Ensure df is None if problematic
                        elif df.empty:
                            logger.warning(f"{filename} is empty. Marking for re-fetch.")
                            needs_refetch = True
                        else:
                            df.set_index("datetime", inplace=True)
                            original_rows = len(df)
                            # Filter by date range if provided, AFTER loading from cache
                            if start_date and end_date:
                                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                                if df.empty and original_rows > 0 : # Was not empty before date filtering
                                    logger.warning(f"{filename} has no data in requested range ({start_date} to {end_date}). Marking for re-fetch.")
                                    needs_refetch = True # If filtering makes it empty, and it wasn't originally, consider re-fetching
                                    df = None
                            
                            if not needs_refetch and df is not None and not df.empty:
                                # Basic sanitisation: drop rows where any OHLC column is 0/NaN
                                required_cols = ['open', 'high', 'low', 'close']
                                if all(col in df.columns for col in required_cols):
                                    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
                                    before_sanitize_rows = len(df)
                                    df = df.dropna(subset=required_cols)
                                    df = df[(df[required_cols] != 0).all(axis=1)]
                                    after_sanitize_rows = len(df)
                                    if after_sanitize_rows == 0 and before_sanitize_rows > 0:
                                        logger.warning(f"{filename} contained only zero/NaN OHLC rows after sanitisation. Marking for re-fetch.")
                                        needs_refetch = True
                                        df = None
                                    elif after_sanitize_rows < before_sanitize_rows:
                                        logger.info(f"Sanitised {filename}: removed {before_sanitize_rows - after_sanitize_rows} bad rows (now {after_sanitize_rows}).")
                                else:
                                    logger.warning(f"{filename} missing critical OHLC columns. Marking for re-fetch.")
                                    needs_refetch = True
                                    df = None
                                
                                if not needs_refetch and df is not None and not df.empty:
                                    data[symbol][tf] = df
                                    logger.info(f"Loaded {len(df)} rows for {symbol} {tf} from {filename} (Date range: {df.index.min()} to {df.index.max() if not df.empty else 'N/A'})")
                                elif df is None or df.empty: # If sanitization or date filter emptied it, ensure needs_refetch is true
                                    needs_refetch = True


                    except Exception as e:
                        logger.warning(f"Failed to load or process {filename}: {e}. Marking for re-fetch.")
                        needs_refetch = True
                        df = None 
                else:
                    logger.info(f"Local file {filename} not found. Marking for potential fetch.")
                    needs_refetch = True
                
                if needs_refetch:
                    if not allow_external_fetch:
                        logger.warning(f"Data for {symbol} {tf} needs refetch, but external fetching is disabled. No data loaded.")
                        data[symbol][tf] = pd.DataFrame()
                        continue # To the next timeframe or symbol

                    logger.info(f"Attempting to fetch data for {symbol} {tf} from external source.")
                    current_mt5_handler = mt5_handler_instance
                    if current_mt5_handler is None:
                        logger.info("MT5Handler instance not provided, attempting to initialize one for fetching.")
                        try:
                            current_mt5_handler = MT5Handler.get_instance()
                        except Exception as e: # Catch errors during MT5Handler.get_instance()
                            logger.error(f"Failed to initialize MT5Handler: {e}. Cannot fetch data for {symbol} {tf}.")
                            data[symbol][tf] = pd.DataFrame()
                            continue 
                    
                    if current_mt5_handler is None or not current_mt5_handler.initialized: # Use .initialized property
                        logger.warning(f"MT5 handler not available or not initialized. Cannot fetch {symbol} {tf}.")
                        data[symbol][tf] = pd.DataFrame()
                        continue

                    try:
                        fetched_df = None
                        if start_date and end_date:
                            fetched_df = current_mt5_handler.get_historical_data(symbol, tf, pd.to_datetime(start_date), pd.to_datetime(end_date))
                        else:
                            fetched_df = current_mt5_handler.get_market_data(symbol, tf, num_bars)
                        
                        if fetched_df is not None and not fetched_df.empty:
                            # Ensure 'datetime' is a column for saving and indexing
                            if 'datetime' not in fetched_df.columns:
                                if fetched_df.index.name == "datetime" or fetched_df.index.name == "time":
                                    fetched_df = fetched_df.reset_index().rename(columns={fetched_df.index.name: 'datetime'})
                                else: # Fallback: try to create a datetime column from index if name is None or other
                                    logger.warning(f"Index for {symbol} {tf} fetched data is not named 'datetime' or 'time'. Using index as is for 'datetime' column.")
                                    fetched_df['datetime'] = fetched_df.index 
                            
                            # Convert datetime column to pandas datetime objects if not already
                            fetched_df['datetime'] = pd.to_datetime(fetched_df['datetime'])

                            # Save to CSV with 'datetime' as a column
                            fetched_df.to_csv(filename, index=False) 
                            
                            # Set index for in-memory df
                            fetched_df.set_index("datetime", inplace=True)

                            # Final sanitisation identical to cache path
                            required_cols = ['open', 'high', 'low', 'close']
                            if all(col in fetched_df.columns for col in required_cols):
                                fetched_df[required_cols] = fetched_df[required_cols].apply(pd.to_numeric, errors='coerce')
                                before_sanitize_rows = len(fetched_df)
                                fetched_df = fetched_df.dropna(subset=required_cols)
                                fetched_df = fetched_df[(fetched_df[required_cols] != 0).all(axis=1)]
                                if len(fetched_df) < before_sanitize_rows:
                                    logger.info(f"Sanitised freshly fetched {symbol} {tf}: removed {before_sanitize_rows - len(fetched_df)} bad rows (now {len(fetched_df)}).")
                            
                            if fetched_df.empty:
                                logger.warning(f"Fetched data for {symbol} {tf} became empty after sanitization. No data loaded.")
                                data[symbol][tf] = pd.DataFrame()
                        else:
                            logger.warning(f"Fetched data for {symbol} {tf} was None or empty after sanitization. No data loaded.")
                            data[symbol][tf] = pd.DataFrame()
                    except Exception as e:
                        logger.error(f"Failed to fetch or process {symbol} {tf} from MT5: {e}")
                        data[symbol][tf] = pd.DataFrame()
                elif df is not None: # If not needs_refetch and df was loaded successfully
                    data[symbol][tf] = df
                else: # Should not happen if logic is correct, but as a fallback
                    logger.warning(f"Data for {symbol} {tf} could not be loaded or fetched. Assigning empty DataFrame.")
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
    config = kwargs.get('config', {}) # Get config, default to empty dict
    allow_external_fetch = config.get('allow_external_fetch', True) # Default to True if not in config

    if mode == 'csv':
        directory = kwargs.get('directory')
        if not directory:
            raise ValueError("CSV data loader requires 'directory' argument.")
        # CSV loader doesn't fetch externally, so allow_external_fetch is less relevant here
        # but we keep the signature consistent for the lambda if it were to be used by load_or_fetch.
        return lambda symbols, timeframes: Backtester.load_csv_data(directory, symbols, timeframes)
    elif mode == 'mt5':
        mt5_handler = kwargs.get('mt5_handler')
        num_bars = kwargs.get('num_bars', 1000)
        if not mt5_handler:
             # If direct MT5 mode is chosen but no handler, this implies an issue or intentional offline use for this mode too.
            if not allow_external_fetch:
                logger.warning("MT5 mode selected, but external fetching is disabled and no MT5 handler provided. Will return empty data.")
                return lambda symbols, timeframes: {s: {tf: pd.DataFrame() for tf in timeframes} for s in symbols}
            else:
                # Try to get an instance if allow_external_fetch is True
                logger.info("MT5 handler not provided for 'mt5' mode, attempting to get a new instance.")
                try:
                    mt5_handler = MT5Handler.get_instance()
                    if not mt5_handler.initialized:
                        raise ConnectionError("MT5 handler could not be initialized.")
                except Exception as e:
                    logger.error(f"Failed to initialize MT5Handler for 'mt5' mode: {e}. Returning empty data structure.")
                    return lambda symbols, timeframes: {s: {tf: pd.DataFrame() for tf in timeframes} for s in symbols}

        return lambda symbols, timeframes: Backtester.load_mt5_data(mt5_handler, symbols, timeframes, start_date=start_date, end_date=end_date, num_bars=num_bars)
    elif mode == 'cache':
        directory = kwargs.get('directory')
        mt5_handler = kwargs.get('mt5_handler') # This can be None, load_or_fetch will handle it
        num_bars = kwargs.get('num_bars', 1000)
        if not directory:
            raise ValueError("Cache data loader requires 'directory' argument.")
        return lambda symbols, timeframes: Backtester.load_or_fetch_ohlcv_data(
            directory, 
            mt5_handler, 
            symbols, 
            timeframes, 
            start_date=start_date, 
            end_date=end_date, 
            num_bars=num_bars, 
            allow_external_fetch=allow_external_fetch
        )
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
            performance_tracker=cfg.get('performance_tracker'),
            export_config=cfg.get('export_config', {})
        )
        # Run backtest
        backtester.run()
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