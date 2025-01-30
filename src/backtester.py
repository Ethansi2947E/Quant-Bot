import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import MetaTrader5 as mt5
import plotly.graph_objects as go
from pathlib import Path
import json
import os

from config.config import BACKTEST_CONFIG, TRADING_CONFIG, MT5_CONFIG
from src.models import Trade, Signal
from src.signal_generator import SignalGenerator
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis

class Backtester:
    def __init__(self, config=None):
        """Initialize the backtester with configuration."""
        self.config = config or BACKTEST_CONFIG
        self.trading_config = TRADING_CONFIG
        self.balance = self.config["initial_balance"]
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        # Initialize MT5
        if not self.initialize_mt5():
            raise RuntimeError("Failed to initialize MT5")
        
        # Initialize analysis components
        self.signal_generator = SignalGenerator()
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
        
        # Create results directory
        Path(self.config["results_dir"]).mkdir(parents=True, exist_ok=True)
        
        self.data_cache_dir = "data_cache"
        os.makedirs(self.data_cache_dir, exist_ok=True)
        self.cached_data = {}
    
    def initialize_mt5(self) -> bool:
        """Initialize connection to MetaTrader 5."""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Login to MT5
            if not mt5.login(
                login=MT5_CONFIG["login"],
                password=MT5_CONFIG["password"],
                server=MT5_CONFIG["server"]
            ):
                logger.error("MT5 login failed")
                return False
            
            logger.info("MT5 initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False
            
    def __del__(self):
        """Cleanup MT5 connection on object destruction."""
        mt5.shutdown()
    
    def _get_cache_filename(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generate a unique cache filename based on parameters."""
        # Format dates to be filename-safe
        start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else start_date.replace("-", "")
        end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date.replace("-", "")
        return f"{self.data_cache_dir}/{symbol}_{timeframe}_{start_str}_{end_str}.csv"
        
    def _load_cached_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data for {symbol} {timeframe}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return None
        
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str, start_date: str, end_date: str):
        """Save data to cache."""
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        df.to_csv(cache_file)
        logger.info(f"Saved data to cache for {symbol} {timeframe}")

    def get_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data with caching."""
        # Try to load from cache first
        cached_data = self._load_cached_data(symbol, timeframe, start_date, end_date)
        if cached_data is not None and not cached_data.empty and len(cached_data) > 10:  # Added validation
            return cached_data
            
        # If not in cache or invalid cache, download and save to cache
        logger.info(f"Downloading fresh data for {symbol} {timeframe}")
        data = self._download_historical_data(symbol, timeframe, start_date, end_date)
        if not data.empty:
            self._save_to_cache(data, symbol, timeframe, start_date, end_date)
        return data
    
    def run_backtest(self) -> Dict:
        """Run backtest over the specified period."""
        # Handle both string and datetime inputs
        if isinstance(self.config["start_date"], str):
            start_date = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        else:
            start_date = self.config["start_date"]
            
        if isinstance(self.config["end_date"], str):
            end_date = datetime.strptime(self.config["end_date"], "%Y-%m-%d")
        else:
            end_date = self.config["end_date"]
            
        results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        for symbol in self.config["symbols"]:
            symbol_results = {tf: {} for tf in self.config["timeframes"]}
            
            for timeframe in self.config["timeframes"]:
                # Get historical data
                data = self.get_historical_data(symbol, timeframe, start_date, end_date)
                if data.empty:
                    continue
                
                # Run analysis and generate signals
                signals = self.analyze_market_data(data, symbol, timeframe)
                
                # Simulate trades
                trades = self.simulate_trades(signals, data, symbol)
                results['trades'].extend(trades)
                
                # Calculate metrics
                metrics = self.calculate_metrics(trades)
                symbol_results[timeframe] = metrics
            
            results['metrics'][symbol] = symbol_results
        
        # Save results
        if self.config["save_results"]:
            self.save_results(results)
        
        # Generate visualization
        if self.config["enable_visualization"]:
            self.visualize_results()
        
        # Calculate final metrics
        profits = [t.pnl for t in results['trades']]
        results['max_drawdown'] = self.calculate_max_drawdown()
        results['sharpe_ratio'] = self.calculate_sharpe_ratio(profits)
        
        # Print summary
        self._print_backtest_summary(results)
        
        return results
    
    def analyze_market_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Analyze market data and generate signals."""
        signals = []
        window_size = 100  # Process data in chunks for efficiency
        
        # Pre-calculate indicators for entire dataset
        data = self.signal_generator.calculate_indicators(data.copy())
        
        # Process data in chunks to improve performance
        for i in range(window_size, len(data), 20):  # Step by 20 candles
            current_data = data.iloc[max(0, i-window_size):i+1]
            
            # Generate signal using current data
            signal = self.signal_generator.generate_signal(
                df=current_data,
                symbol=symbol,
                timeframe=timeframe,
                mtf_data={"current": current_data}
            )
            
            # Only append actionable signals (not HOLD)
            if signal and signal.get('direction') and signal.get('direction') != 'HOLD':
                # Only log actionable signals
                logger.info(f"Generated {signal['direction']} signal for {symbol} ({timeframe})")
                logger.info(f"Entry: {signal.get('entry_price')}, SL: {signal.get('stop_loss')}, TP: {signal.get('take_profit')}")
                signals.append(signal)
        
        logger.info(f"Analysis complete for {symbol} {timeframe}. Found {len(signals)} actionable signals.")
        return signals
    
    def simulate_trades(self, signals: List[Signal], data: pd.DataFrame, symbol: str) -> List[Trade]:
        """Simulate trades based on signals."""
        trades = []
        
        for signal in signals:
            # Skip signals without required fields
            if not all(k in signal for k in ['direction', 'entry_price', 'stop_loss', 'take_profit', 'timestamp']):
                logger.warning(f"Skipping invalid signal: {signal}")
                continue
            
            # Calculate position size based on risk
            risk_amount = self.balance * self.config["risk_per_trade"]
            position_size = self.calculate_position_size(risk_amount, signal['stop_loss'], signal['entry_price'])
            
            # Convert BUY/SELL to LONG/SHORT
            direction = "LONG" if signal['direction'] == "BUY" else "SHORT"
            
            # Create trade
            trade = Trade(
                symbol=symbol,
                direction=direction,
                position_size=position_size,
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                entry_time=signal['timestamp']
            )
            
            # Simulate trade execution
            trade_result = self.simulate_trade_execution(trade, data)
            trades.append(trade_result)
            
            # Update balance
            self.balance += trade_result.pnl if trade_result.pnl else 0
            self.equity_curve.append((trade_result.exit_time, self.balance))
            
            # Log trade result
            logger.info(f"Trade completed: {trade_result.direction} {symbol}")
            logger.info(f"Entry: {trade_result.entry_price:.5f}, Exit: {trade_result.exit_price:.5f}")
            logger.info(f"PnL: {trade_result.pnl:.2f}, New Balance: {self.balance:.2f}")
        
        return trades
    
    def simulate_trade_execution(self, trade: Trade, data: pd.DataFrame) -> Trade:
        """Simulate the execution of a single trade."""
        # Find entry point using index values
        entry_mask = data.index >= trade.entry_time
        if not any(entry_mask):
            logger.warning(f"No data found after entry time {trade.entry_time}")
            trade.exit_price = trade.entry_price  # Set exit same as entry if no data
            trade.exit_time = trade.entry_time
            trade.pnl = 0
            return trade
        
        entry_idx = data[entry_mask].index[0]
        trade_closed = False
        
        for i in range(data.index.get_loc(entry_idx) + 1, len(data)):
            current_bar = data.iloc[i]
            
            if trade.direction == "LONG":
                # Check if stop loss hit
                if current_bar['low'] <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = current_bar.name
                    trade.pnl = self.calculate_profit(trade)
                    trade_closed = True
                    break
                    
                # Check if take profit hit
                if current_bar['high'] >= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_time = current_bar.name
                    trade.pnl = self.calculate_profit(trade)
                    trade_closed = True
                    break
            else:  # SHORT
                # Check if stop loss hit
                if current_bar['high'] >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = current_bar.name
                    trade.pnl = self.calculate_profit(trade)
                    trade_closed = True
                    break
                    
                # Check if take profit hit
                if current_bar['low'] <= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_time = current_bar.name
                    trade.pnl = self.calculate_profit(trade)
                    trade_closed = True
                    break
        
        # If trade never hit SL or TP, close at last available price
        if not trade_closed:
            last_bar = data.iloc[-1]
            trade.exit_price = last_bar['close']
            trade.exit_time = last_bar.name
            trade.pnl = self.calculate_profit(trade)
            logger.info(f"Trade closed at end of data: {trade.direction} {trade.symbol}")
        
        return trade
    
    def calculate_position_size(self, risk_amount: float, stop_loss: float, entry_price: float) -> float:
        """Calculate position size based on risk parameters."""
        pip_value = 0.0001  # For most forex pairs
        pips_at_risk = abs(entry_price - stop_loss) / pip_value
        position_size = risk_amount / pips_at_risk
        return position_size
    
    def calculate_profit(self, trade: Trade) -> float:
        """Calculate profit/loss for a trade including commission."""
        pip_value = 0.0001
        pips = (trade.exit_price - trade.entry_price) / pip_value if trade.direction == "LONG" else (trade.entry_price - trade.exit_price) / pip_value
        profit = pips * trade.position_size * 10  # Assuming standard lot size calculation
        
        # Subtract commission
        commission = trade.position_size * self.config["commission"] * 2  # multiply by 2 for entry and exit
        return profit - commission
    
    def calculate_metrics(self, trades: List[Trade]) -> Dict:
        """Calculate trading performance metrics."""
        if not trades:
            return {}
            
        profits = [t.pnl for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        metrics = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0,
            "total_profit": sum(profits),
            "average_profit": np.mean(profits) if profits else 0,
            "max_drawdown": self.calculate_max_drawdown(),
            "profit_factor": abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            "sharpe_ratio": self.calculate_sharpe_ratio(profits),
            "max_consecutive_losses": self.calculate_max_consecutive_losses(profits)
        }
        
        return metrics
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0
            
        equity = [e[1] for e in self.equity_curve]
        peak = equity[0]
        max_dd = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio of returns."""
        if not profits:
            return 0
            
        returns = pd.Series(profits) / self.config["initial_balance"]
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        
        if excess_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_max_consecutive_losses(self, profits: List[float]) -> int:
        """Calculate maximum consecutive losing trades."""
        max_losses = current_losses = 0
        
        for profit in profits:
            if profit <= 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
                
        return max_losses
    
    def visualize_results(self):
        """Create visualization of backtest results."""
        if not self.equity_curve:
            return
            
        dates = [e[0] for e in self.equity_curve]
        equity = [e[1] for e in self.equity_curve]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity Curve'
        ))
        
        fig.update_layout(
            title='Backtest Results - Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity',
            template='plotly_dark'
        )
        
        fig.write_html(f"{self.config['results_dir']}/equity_curve.html")
    
    def save_results(self, results: Dict):
        """Save backtest results to file with detailed information."""
        # Create a more descriptive timestamp with symbol and date range
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_str = '_'.join(self.config["symbols"])
        date_range = f"{self.config['start_date'].replace('-', '')}_{self.config['end_date'].replace('-', '')}"
        timeframe_str = '_'.join(self.config["timeframes"])
        file_prefix = f"backtest_{symbol_str}_{timeframe_str}_{date_range}"
        
        # Save JSON results
        json_filepath = f"{self.config['results_dir']}/{file_prefix}.json"
        detailed_results = {
            "symbols": self.config["symbols"],
            "timeframes": self.config["timeframes"],
            "results": results
        }
        with open(json_filepath, 'w') as f:
            json.dump(detailed_results, f, indent=4, default=str)
        
        # Save Excel results
        try:
            import pandas as pd
            excel_filepath = f"{self.config['results_dir']}/{file_prefix}.xlsx"
            
            # Format the DataFrames before writing to Excel
            def format_numeric_columns(df):
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['rate', 'ratio']):
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, (int, float)) else x)
                    elif any(term in col_lower for term in ['pnl', 'profit', 'loss', 'balance', 'risk', 'reward', '$']):
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, (int, float)) else x)
                return df

            # Prepare data structures
            summary_data = {
                'Metric': ['Symbols', 'Timeframes', 'Period', 'Initial Balance', 'Final Balance', 'Total Trades', 'Winning Trades', 
                          'Losing Trades', 'Win Rate', 'Total PnL', 'Average Win', 'Average Loss', 'Largest Win', 'Largest Loss', 
                          'Profit Factor', 'Sharpe Ratio', 'Max Drawdown', 'Average RR Ratio', 'SL Hit Rate', 'TP Hit Rate'],
                'Value': [
                    ', '.join(self.config['symbols']),
                    ', '.join(self.config['timeframes']),
                    f"{self.config['start_date']} to {self.config['end_date']}",
                    self.config['initial_balance'],
                    self.equity_curve[-1][1] if self.equity_curve else self.config['initial_balance'],  # Final balance from equity curve
                    len(results['trades']),
                    sum(1 for t in results['trades'] if t.pnl > 0),
                    sum(1 for t in results['trades'] if t.pnl <= 0),
                    len([t for t in results['trades'] if t.pnl > 0]) / len(results['trades']) if results['trades'] else 0,
                    sum(t.pnl for t in results['trades']),
                    np.mean([t.pnl for t in results['trades'] if t.pnl > 0]) if any(t.pnl > 0 for t in results['trades']) else 0,
                    np.mean([t.pnl for t in results['trades'] if t.pnl <= 0]) if any(t.pnl <= 0 for t in results['trades']) else 0,
                    max((t.pnl for t in results['trades']), default=0),
                    min((t.pnl for t in results['trades']), default=0),
                    abs(sum(t.pnl for t in results['trades'] if t.pnl > 0) / sum(t.pnl for t in results['trades'] if t.pnl <= 0)) if sum(t.pnl for t in results['trades'] if t.pnl <= 0) != 0 else float('inf'),
                    results.get('sharpe_ratio', 0),
                    results.get('max_drawdown', 0),
                    results.get('avg_rr', 0),
                    sum(1 for t in results['trades'] if t.exit_price == t.stop_loss) / len(results['trades']) if results['trades'] else 0,
                    sum(1 for t in results['trades'] if t.exit_price == t.take_profit) / len(results['trades']) if results['trades'] else 0
                ]
            }

            trades_data = [{
                'Trade #': i+1,
                'Symbol': t.symbol,
                'Direction': t.direction,
                'Entry Time': t.entry_time,
                'Exit Time': t.exit_time,
                'Entry Price': t.entry_price,
                'Exit Price': t.exit_price,
                'Stop Loss': t.stop_loss,
                'Take Profit': t.take_profit,
                'Position Size': t.position_size,
                'Risk Amount': abs(t.entry_price - t.stop_loss) * t.position_size * 10000,
                'PnL': t.pnl,
                'Outcome': 'SL Hit' if t.exit_price == t.stop_loss else 'TP Hit' if t.exit_price == t.take_profit else 'Closed'
            } for i, t in enumerate(results['trades'])]

            time_analysis_data = results.get('time_analysis', {'Hour': [], 'Trades': [], 'Win Rate': [], 'Total PnL': []})
            daily_analysis_data = results.get('daily_analysis', {'Day': [], 'Trades': [], 'Win Rate': [], 'Total PnL': []})
            consecutive_data = results.get('consecutive_analysis', {'Metric': [], 'Value': []})
            risk_data = results.get('risk_analysis', {'Metric': [], 'Value': []})

            # Format each DataFrame
            summary_df = format_numeric_columns(pd.DataFrame(summary_data))
            trades_df = format_numeric_columns(pd.DataFrame(trades_data))
            time_df = format_numeric_columns(pd.DataFrame(time_analysis_data))
            daily_df = format_numeric_columns(pd.DataFrame(daily_analysis_data))
            consecutive_df = format_numeric_columns(pd.DataFrame(consecutive_data))
            risk_df = format_numeric_columns(pd.DataFrame(risk_data))

            # Write to Excel with float_format for proper number formatting
            with pd.ExcelWriter(excel_filepath, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Add formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D9D9D9',
                    'border': 1,
                    'text_wrap': True,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                num_format = workbook.add_format({'num_format': '#,##0.00'})
                percent_format = workbook.add_format({'num_format': '0.00%'})
                price_format = workbook.add_format({'num_format': '0.00000'})
                
                # Write each sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Only write trade-related sheets if there are trades
                if len(results['trades']) > 0:
                    trades_df.to_excel(writer, sheet_name='Detailed Trades', index=False)
                    time_df.to_excel(writer, sheet_name='Time Analysis', index=False)
                    daily_df.to_excel(writer, sheet_name='Daily Analysis', index=False)
                    consecutive_df.to_excel(writer, sheet_name='Consecutive Analysis', index=False)
                    risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
                else:
                    # Create empty sheets with headers for consistency
                    pd.DataFrame(columns=['No trades executed']).to_excel(writer, sheet_name='Detailed Trades', index=False)
                    pd.DataFrame(columns=['No trades executed']).to_excel(writer, sheet_name='Time Analysis', index=False)
                    pd.DataFrame(columns=['No trades executed']).to_excel(writer, sheet_name='Daily Analysis', index=False)
                    pd.DataFrame(columns=['No trades executed']).to_excel(writer, sheet_name='Consecutive Analysis', index=False)
                    pd.DataFrame(columns=['No trades executed']).to_excel(writer, sheet_name='Risk Analysis', index=False)
                
                # Format each sheet
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column('A:Z', 15)
                    
                    # Get the dataframe corresponding to the current sheet
                    if sheet_name == 'Summary':
                        df = summary_df
                    elif len(results['trades']) > 0:  # Only use detailed dataframes if there are trades
                        if sheet_name == 'Detailed Trades':
                            df = trades_df
                        elif sheet_name == 'Time Analysis':
                            df = time_df
                        elif sheet_name == 'Daily Analysis':
                            df = daily_df
                        elif sheet_name == 'Consecutive Analysis':
                            df = consecutive_df
                        else:  # Risk Analysis
                            df = risk_df
                    else:
                        # For empty sheets, use the simple "No trades" DataFrame
                        df = pd.DataFrame(columns=['No trades executed'])
                    
                    # Apply header format
                    for col_num, value in enumerate(df.columns):
                        worksheet.write(0, col_num, str(value), header_format)
                    
                    # Only process data rows if there are any
                    if len(df) > 0:
                        for row in range(len(df)):
                            for col in range(len(df.columns)):
                                value = df.iloc[row, col]
                                if pd.isna(value):  # Handle NaN values
                                    worksheet.write(row + 1, col, '')
                                    continue
                                    
                                col_name = df.columns[col].lower()
                                try:
                                    if isinstance(value, (int, float)):
                                        if any(term in col_name for term in ['rate', 'ratio']):
                                            worksheet.write_number(row + 1, col, float(value), percent_format)
                                        elif any(term in col_name for term in ['pnl', 'profit', 'loss', 'balance', 'risk', 'reward', '$']):
                                            worksheet.write_number(row + 1, col, float(value), num_format)
                                        elif 'price' in col_name:
                                            worksheet.write_number(row + 1, col, float(value), price_format)
                                        else:
                                            worksheet.write_number(row + 1, col, float(value))
                                    else:
                                        worksheet.write(row + 1, col, str(value))
                                except (ValueError, TypeError):
                                    worksheet.write(row + 1, col, str(value))
                    
                    # Apply autofilter only if there are columns
                    if len(df.columns) > 0:
                        worksheet.autofilter(0, 0, max(len(df), 0), len(df.columns) - 1)
                
                logger.info(f"Backtest results saved to Excel file: {excel_filepath}")
                logger.info(f"Backtest results saved to JSON file: {json_filepath}")
                
        except Exception as e:
            logger.error(f"Error saving Excel results: {str(e)}")
    
    def _print_backtest_summary(self, results: Dict) -> None:
        """Print a summary of backtest results."""
        logger.info("\n=== BACKTEST SUMMARY ===")
        
        # Display timeframe and symbol information prominently
        logger.info("\nBacktest Configuration:")
        logger.info(f"Timeframe: {', '.join(self.config['timeframes'])}")
        logger.info(f"Symbol(s): {', '.join(self.config['symbols'])}")
        logger.info(f"Period: {self.config['start_date']} to {self.config['end_date']}")
        
        # Overall Statistics
        trades = results.get('trades', [])
        total_trades = len(trades)
        if total_trades == 0:
            logger.info("No trades executed during backtest period")
            return
        
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate PnL statistics
        pnl_values = [t.pnl for t in trades]
        total_pnl = sum(pnl_values)
        avg_win = sum([pnl for pnl in pnl_values if pnl > 0]) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum([pnl for pnl in pnl_values if pnl <= 0]) / losing_trades if losing_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = results.get('max_drawdown', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        
        # Print summary
        logger.info(f"\nTrading Statistics:")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"\nProfitability:")
        logger.info(f"Total PnL: {total_pnl:.2f}")
        logger.info(f"Average Win: {avg_win:.2f}")
        logger.info(f"Average Loss: {abs(avg_loss):.2f}")
        logger.info(f"Profit Factor: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Profit Factor: ∞")
        logger.info(f"\nRisk Metrics:")
        logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Detailed Trade Analysis
        logger.info("\n=== DETAILED TRADE ANALYSIS ===")
        logger.info("\nTrade-by-Trade Breakdown:")
        logger.info(f"| {'#':<4} | {'Entry Time':<19} | {'Exit Time':<19} | {'Dir':<4} | {'Entry':<10} | {'Exit':<10} | {'SL':<10} | {'TP':<10} | {'Risk($)':<8} | {'Outcome':<8} | {'PnL($)':<10} |")
        logger.info("-" * 140)
        
        for i, trade in enumerate(trades, 1):
            sl_hit = trade.exit_price == trade.stop_loss
            tp_hit = trade.exit_price == trade.take_profit
            outcome = "SL Hit" if sl_hit else "TP Hit" if tp_hit else "Closed"
            risk_amount = abs(trade.entry_price - trade.stop_loss) * trade.position_size * 10000
            
            # Format timestamps
            entry_time = trade.entry_time.strftime("%Y-%m-%d %H:%M")
            exit_time = trade.exit_time.strftime("%Y-%m-%d %H:%M")
            
            logger.info(
                f"| {i:<4} | {entry_time:<19} | {exit_time:<19} | {trade.direction[0]:<4} | "
                f"{trade.entry_price:.5f} | {trade.exit_price:.5f} | {trade.stop_loss:.5f} | "
                f"{trade.take_profit:.5f} | {risk_amount:.2f} | {outcome:<8} | {trade.pnl:.2f} |"
            )
        
        # Print trade distribution by symbol
        logger.info("\nTrade Distribution by Symbol:")
        symbol_trades = {}
        for trade in trades:
            symbol = trade.symbol
            symbol_trades[symbol] = symbol_trades.get(symbol, 0) + 1
        for symbol, count in symbol_trades.items():
            logger.info(f"{symbol}: {count} trades ({(count/total_trades)*100:.1f}%)")
        
        # Print performance by timeframe
        logger.info("\nPerformance by Symbol and Timeframe:")
        for symbol, timeframes in results['metrics'].items():
            logger.info(f"\n{symbol}:")
            for timeframe, metrics in timeframes.items():
                if metrics:
                    logger.info(f"  {timeframe}:")
                    logger.info(f"    Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
                    logger.info(f"    Total Profit: {metrics.get('total_profit', 0):.2f}")
                    logger.info(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        
        # Additional Statistics
        sl_hits = sum(1 for t in trades if t.exit_price == t.stop_loss)
        tp_hits = sum(1 for t in trades if t.exit_price == t.take_profit)
        other_closes = total_trades - sl_hits - tp_hits
        
        logger.info("\nTrade Exit Analysis:")
        logger.info(f"Stop Loss Hits: {sl_hits} ({(sl_hits/total_trades)*100:.1f}%)")
        logger.info(f"Take Profit Hits: {tp_hits} ({(tp_hits/total_trades)*100:.1f}%)")
        logger.info(f"Other Closes: {other_closes} ({(other_closes/total_trades)*100:.1f}%)")
        
        logger.info("\n=== END OF SUMMARY ===\n")

    def _download_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download historical data from MT5."""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5, 
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        logger.info(f"Downloading data for {symbol} {timeframe}")
        rates = mt5.copy_rates_range(symbol, timeframe_map[timeframe], start_date, end_date)
        if rates is None:
            logger.error(f"Failed to get historical data for {symbol} {timeframe}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df['time'] = df.index  # Keep time as both index and column
        
        # Add tick volume as volume
        if 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
        
        return df 