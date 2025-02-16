import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from loguru import logger
import MetaTrader5 as mt5
import plotly.graph_objects as go
from pathlib import Path
import json
import os
import sys

# Configure logger for better formatting
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

from config.config import BACKTEST_CONFIG, TRADING_CONFIG, MT5_CONFIG
from src.models import Trade, Signal
from src.signal_generator import SignalGenerator
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis
from src.risk_manager import RiskManager
from src.common import trading_logic

class Backtester:
    def __init__(self, config=None):
        """Initialize the backtester with configuration."""
        self.config = config or BACKTEST_CONFIG
        self.trading_config = TRADING_CONFIG
        self.balance = self.config["initial_balance"]
        self.risk_per_trade = self.config.get("risk_per_trade", 0.01)  # Default to 1% if not specified
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        # Initialize MT5
        if not self.initialize_mt5():
            raise RuntimeError("Failed to initialize MT5")
        
        # Initialize analysis components
        self.risk_manager = RiskManager()  # Initialize RiskManager
        self.signal_generator = SignalGenerator()
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
        
        # Create results directory and subdirectories
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "trades").mkdir(exist_ok=True)
        (self.results_dir / "analysis").mkdir(exist_ok=True)
        (self.results_dir / "charts").mkdir(exist_ok=True)
        
        self.data_cache_dir = "data_cache"
        os.makedirs(self.data_cache_dir, exist_ok=True)
        self.cached_data = {}
        self.trading_logic = trading_logic.TradingLogic(self.config)
    
    def initialize_mt5(self) -> bool:
        """Initialize connection to MetaTrader 5."""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.warning("MT5 initialization failed, will use cached data if available")
                return True  # Return True to allow backtesting with cached data
            
            # Login to MT5
            if not mt5.login(
                login=MT5_CONFIG["login"],
                password=MT5_CONFIG["password"],
                server=MT5_CONFIG["server"]
            ):
                logger.warning("MT5 login failed, will use cached data if available")
                return True  # Return True to allow backtesting with cached data
            
            logger.info("MT5 initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Error initializing MT5: {str(e)}, will use cached data if available")
            return True  # Return True to allow backtesting with cached data
            
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
        try:
            # Handle both string and datetime inputs
            if isinstance(self.config["start_date"], str):
                start_date = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
            else:
                start_date = self.config["start_date"]
                
            if isinstance(self.config["end_date"], str):
                end_date = datetime.strptime(self.config["end_date"], "%Y-%m-%d")
            else:
                end_date = self.config["end_date"]
                
            all_results = {}
            
            # Run separate backtest for each symbol
            for symbol in self.config["symbols"]:
                try:
                    logger.info(f"\n=== Starting backtest for {symbol} ===")
                    
                    # Reset balance and equity curve for each symbol
                    self.balance = self.config["initial_balance"]
                    self.equity_curve = []
                    self.trades = []
                    
                    symbol_results = {
                        'trades': [],
                        'equity_curve': [],
                        'metrics': {}
                    }
                    
                    # Process each timeframe
                    for timeframe in self.config["timeframes"]:
                        try:
                            # Get historical data
                            data = self.get_historical_data(symbol, timeframe, start_date, end_date)
                            if data.empty:
                                logger.warning(f"No data available for {symbol} {timeframe}")
                                continue
                            
                            # Run analysis and generate signals using common trading logic
                            signals = self.trading_logic.analyze_market_data(data, symbol, timeframe)
                            if not signals:
                                logger.warning(f"No signals generated for {symbol} {timeframe}")
                                continue
                            
                            # Simulate trades using common trading logic
                            trades = self.trading_logic.simulate_trades(signals, data, symbol)
                            if trades:
                                symbol_results['trades'].extend(trades)
                                logger.info(f"Generated {len(trades)} trades for {symbol} {timeframe}")
                            else:
                                logger.warning(f"No trades generated for {symbol} {timeframe}")
                            
                        except Exception as e:
                            logger.error(f"Error processing timeframe {timeframe} for {symbol}: {str(e)}")
                            continue
                    
                    # Process results if we have trades
                    if symbol_results.get('trades'):
                        try:
                            logger.info(f"Processing results for symbol: {symbol}")
                            logger.debug(f"Trades count for {symbol}: {len(symbol_results['trades'])}")
                            
                            # Calculate metrics for all trades
                            try:
                                logger.debug(f"Calculating metrics for {len(symbol_results['trades'])} trades")
                                metrics = self.calculate_metrics(symbol_results['trades'])
                                symbol_results['metrics'] = metrics
                                logger.info(f"Metrics calculated for {symbol}: {metrics}")
                            except Exception as e:
                                logger.exception(f"Error calculating metrics for {symbol}")
                                continue
                            
                            # Generate equity curve data
                            try:
                                running_balance = self.config.get('initial_balance', 100000)
                                equity_curve_data = []
                                
                                # Sort trades by exit time to ensure chronological order
                                sorted_trades = sorted(symbol_results['trades'], key=lambda x: pd.to_datetime(x['exit_time']))
                                
                                for trade in sorted_trades:
                                    running_balance += trade['pnl']
                                    equity_curve_data.append({
                                        'timestamp': trade['exit_time'],
                                        'balance': running_balance
                                    })
                                
                                symbol_results['equity_curve'] = equity_curve_data
                                logger.info(f"Generated equity curve with {len(equity_curve_data)} points for {symbol}")
                            except Exception as e:
                                logger.warning(f"Error generating equity curve for {symbol}: {str(e)}")
                                symbol_results['equity_curve'] = []
                            
                            # Save results if enabled
                            if self.config.get('save_results', False):
                                try:
                                    logger.info(f"Saving results for symbol: {symbol}")
                                    self._save_symbol_results(symbol_results, symbol)
                                except Exception as e:
                                    logger.exception(f"Error saving results for {symbol}")
                            
                            # Visualize if enabled
                            if self.config.get('enable_visualization', False):
                                try:
                                    logger.info(f"Creating visualization for symbol: {symbol}")
                                    self._visualize_symbol_results(symbol_results, symbol)
                                except Exception as e:
                                    logger.exception(f"Error visualizing results for {symbol}")
                            
                            # Print summary
                            try:
                                logger.info(f"Printing backtest summary for symbol: {symbol}")
                                self._print_backtest_summary(symbol_results, symbol)
                            except Exception as e:
                                logger.exception(f"Error printing summary for {symbol}")
                            
                            # Store results in all_results
                            all_results[symbol] = symbol_results
                            logger.info(f"Successfully processed {symbol} with {len(symbol_results['trades'])} trades")
                            
                        except Exception as e:
                            logger.exception(f"Error processing results for {symbol}")
                            logger.debug(f"symbol_results: {symbol_results}")
                            continue
                    else:
                        logger.warning(f"No trades generated for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}")
                    continue
            
            if not all_results:
                logger.warning("No trades generated for any symbol")
            else:
                logger.info(f"Backtest completed successfully for {len(all_results)} symbols")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            return {}
    
    def _calculate_trade_metrics(self, trade: Dict) -> Dict:
        """Calculate detailed metrics for a trade."""
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        
        # Calculate holding time
        holding_time = exit_time - entry_time
        holding_time_hours = holding_time.total_seconds() / 3600
        
        # Calculate R multiple
        risk = abs(trade['entry_price'] - trade['stop_loss'])
        actual_profit = trade['exit_price'] - trade['entry_price']
        r_multiple = actual_profit / risk if risk != 0 else 0
        
        # Add metrics to trade
        trade['holding_time_hours'] = holding_time_hours
        trade['r_multiple'] = r_multiple
        
        return trade
    
    def _save_trade_details(self, trade: Dict, symbol: str):
        """Save detailed trade information to JSON file."""
        trade_data = {
            'id': trade.get('id'),
            'symbol': symbol,
            'entry_time': trade.get('entry_time').strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': trade.get('exit_time', '').strftime('%Y-%m-%d %H:%M:%S') if trade.get('exit_time') else None,
            'trade_type': trade.get('direction'),
            'entry_price': trade.get('entry_price'),
            'exit_price': trade.get('exit_price'),
            'stop_loss': trade.get('stop_loss'),
            'take_profit': trade.get('take_profit'),
            'position_size': trade.get('position_size'),
            'risk_amount': trade.get('risk_amount'),
            'profit_loss': trade.get('pnl'),
            'r_multiple': trade.get('r_multiple'),
            'entry_reason': trade.get('entry_reason'),
            'exit_reason': trade.get('exit_reason'),
            'market_conditions': {
                'trend': trade.get('market_conditions', {}).get('trend'),
                'volatility': trade.get('market_conditions', {}).get('volatility'),
                'session': trade.get('market_conditions', {}).get('session'),
                'atr': trade.get('market_conditions', {}).get('atr'),
                'market_structure': trade.get('market_conditions', {}).get('structure')
            },
            'indicators': {
                'rsi': trade.get('indicators', {}).get('rsi'),
                'macd': trade.get('indicators', {}).get('macd'),
                'volume': trade.get('indicators', {}).get('volume')
            },
            'max_adverse_excursion': trade.get('max_adverse_excursion'),
            'max_favorable_excursion': trade.get('max_favorable_excursion'),
            'holding_time_hours': trade.get('holding_time_hours')
        }
        
        # Save to JSON file
        trade_file = self.results_dir / "trades" / f"{symbol}_{trade['id']}.json"
        with open(trade_file, 'w') as f:
            json.dump(trade_data, f, indent=4)

    def _save_symbol_analysis(self, analysis: Dict, symbol: str):
        """Save detailed symbol analysis to JSON file."""
        analysis_data = {
            'symbol': symbol,
            'period': {
                'start': self.config['start_date'],
                'end': self.config['end_date']
            },
            'overall_metrics': {
                'total_trades': analysis['total_trades'],
                'winning_trades': analysis['winning_trades'],
                'losing_trades': analysis['losing_trades'],
                'win_rate': analysis['win_rate'],
                'profit_factor': analysis['profit_factor'],
                'average_win': analysis['average_win'],
                'average_loss': analysis['average_loss'],
                'largest_win': analysis['largest_win'],
                'largest_loss': analysis['largest_loss'],
                'max_drawdown': analysis['max_drawdown'],
                'sharpe_ratio': analysis['sharpe_ratio'],
                'average_holding_time': analysis['average_holding_time']
            },
            'session_performance': analysis['session_performance'],
            'market_condition_performance': analysis['market_condition_performance'],
            'timeframe_performance': analysis['timeframe_performance'],
            'monthly_performance': analysis['monthly_performance'],
            'risk_metrics': {
                'average_risk_reward': analysis['average_risk_reward'],
                'average_mae': analysis['average_mae'],
                'average_mfe': analysis['average_mfe'],
                'risk_adjusted_return': analysis['risk_adjusted_return']
            }
        }
        
        # Save to JSON file
        analysis_file = self.results_dir / "analysis" / f"{symbol}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=4)
    
    def calculate_position_size(self, risk_amount: float, stop_loss: float, entry_price: float, symbol: str) -> float:
        """Calculate position size using RiskManager."""
        try:
            # Get current session based on time
            current_time = datetime.now().time()
            session = self._determine_session(current_time)
            
            # Use RiskManager's position size calculation
            position_size = self.risk_manager.calculate_position_size(
                account_balance=self.balance,
                risk_amount=risk_amount,
                entry_price=entry_price,
                stop_loss=stop_loss,
                symbol=symbol,
                session=session
            )
            
            logger.info(f"Calculated position size using RiskManager: {position_size} lots (Risk: ${risk_amount:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self.trading_config.get("min_position_size", 0.01)
            
    def _determine_session(self, time: time) -> str:
        """Determine the current trading session based on time."""
        # Convert time to UTC
        hour = time.hour
        
        # Define session times (in UTC)
        if 7 <= hour < 16:  # London session (7:00-16:00 UTC)
            if 13 <= hour < 16:  # London/NY overlap (13:00-16:00 UTC)
                return "london_ny_overlap"
            return "london_open"
        elif 13 <= hour < 22:  # New York session (13:00-22:00 UTC)
            return "ny_open"
        else:  # Asian session
            return "asian"
            
    def _process_trade(self, trade: Dict) -> Dict:
        """Process trade details and ensure all required fields are present."""
        try:
            # Ensure take_profit is present
            if 'take_profit' not in trade:
                stop_distance = abs(trade['entry_price'] - trade['stop_loss'])
                if trade['direction'] == 'BUY':
                    trade['take_profit'] = trade['entry_price'] + (stop_distance * 2)  # 1:2 R:R
                else:
                    trade['take_profit'] = trade['entry_price'] - (stop_distance * 2)  # 1:2 R:R
                    
            # Add other required fields
            trade['id'] = len(self.trades) + 1
            trade['timestamp'] = datetime.now()
            
            return trade
            
        except Exception as e:
            logger.error(f"Error processing trade: {str(e)}")
            return trade
    
    def calculate_profit(self, trade: Trade) -> float:
        """Calculate profit/loss for a trade including commission."""
        try:
            # Calculate pips
            pip_value = 0.0001  # Standard pip value for forex
            if trade.symbol.endswith('JPY'):
                pip_value = 0.01
                
            pips = (trade.exit_price - trade.entry_price) / pip_value if trade.direction == "LONG" else (trade.entry_price - trade.exit_price) / pip_value
            
            # Calculate profit using standard lot size (100,000)
            standard_lot = 100000
            profit = pips * pip_value * standard_lot * trade.position_size
            
            # Subtract commission
            commission = trade.position_size * self.config["commission"] * 2  # multiply by 2 for entry and exit
            return profit - commission
            
        except Exception as e:
            logger.error(f"Error calculating profit: {str(e)}")
            return 0.0
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'total_loss': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'average_holding_time': 0,
                'average_mae': 0,
                'average_mfe': 0,
                'risk_adjusted_return': 0,
                'session_performance': {},
                'market_condition_performance': {},
                'monthly_performance': {}
            }

        try:
            # Basic metrics
            total_trades = len(trades)
            
            # Ensure pnl is a float
            for trade in trades:
                if isinstance(trade['pnl'], (list, tuple)):
                    trade['pnl'] = float(trade['pnl'][0])  # Take first value if it's a list
                else:
                    trade['pnl'] = float(trade['pnl'])  # Convert to float if it's not
            
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            # Profit metrics
            total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            
            average_win = total_profit / len(winning_trades) if winning_trades else 0
            average_loss = total_loss / len(losing_trades) if len(losing_trades) > 0 else 0
            
            largest_win = max((t['pnl'] for t in winning_trades), default=0)
            largest_loss = min((t['pnl'] for t in losing_trades), default=0)
            
            # Calculate drawdown
            equity_curve = []
            running_balance = self.config['initial_balance']
            max_balance = running_balance
            max_drawdown = 0
            
            for trade in trades:
                running_balance += trade['pnl']
                equity_curve.append(running_balance)
                max_balance = max(max_balance, running_balance)
                drawdown = (max_balance - running_balance) / max_balance
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio
            if len(trades) > 1:
                returns = [(t['pnl'] / self.config['initial_balance']) for t in trades]
                avg_return = sum(returns) / len(returns)
                std_dev = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev != 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate average holding time
            holding_times = []
            for trade in trades:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                holding_time = (exit_time - entry_time).total_seconds() / 3600  # Convert to hours
                holding_times.append(holding_time)
            average_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
            
            # Calculate MAE and MFE
            mae_values = [float(t.get('max_adverse_excursion', 0)) for t in trades]
            mfe_values = [float(t.get('max_favorable_excursion', 0)) for t in trades]
            average_mae = sum(mae_values) / len(mae_values) if mae_values else 0
            average_mfe = sum(mfe_values) / len(mfe_values) if mfe_values else 0
            
            # Calculate risk-adjusted return
            risk_adjusted_return = (total_profit - total_loss) / (total_trades * average_mae) if total_trades > 0 and average_mae != 0 else 0
            
            # Session performance
            session_performance = {}
            for trade in trades:
                session = trade.get('session', None)
                if session not in session_performance:
                    session_performance[session] = {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                session_performance[session]['trades'] += 1
                if trade['pnl'] > 0:
                    session_performance[session]['wins'] += 1
                else:
                    session_performance[session]['losses'] += 1
                session_performance[session]['total_pnl'] += trade['pnl']
            
            # Market condition performance
            market_condition_performance = {}
            for trade in trades:
                condition = trade.get('market_condition', {}).get('trend', None)
                if condition not in market_condition_performance:
                    market_condition_performance[condition] = {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                market_condition_performance[condition]['trades'] += 1
                if trade['pnl'] > 0:
                    market_condition_performance[condition]['wins'] += 1
                else:
                    market_condition_performance[condition]['losses'] += 1
                market_condition_performance[condition]['total_pnl'] += trade['pnl']
            
            # Monthly performance
            monthly_performance = {}
            for trade in trades:
                month = pd.to_datetime(trade['entry_time']).strftime('%Y-%m')
                if month not in monthly_performance:
                    monthly_performance[month] = {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                monthly_performance[month]['trades'] += 1
                if trade['pnl'] > 0:
                    monthly_performance[month]['wins'] += 1
                else:
                    monthly_performance[month]['losses'] += 1
                monthly_performance[month]['total_pnl'] += trade['pnl']
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'average_win': average_win,
                'average_loss': average_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'average_holding_time': average_holding_time,
                'average_mae': average_mae,
                'average_mfe': average_mfe,
                'risk_adjusted_return': risk_adjusted_return,
                'session_performance': session_performance,
                'market_condition_performance': market_condition_performance,
                'monthly_performance': monthly_performance
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_metrics: {str(e)}", exc_info=True)
            raise
    
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
                'Risk Amount': t.risk_amount,
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
            
    def _print_backtest_summary(self, results: Dict, symbol: str) -> None:
        """Print a summary of backtest results in a tabular format."""
        trades = results.get('trades', [])
        total_trades = len(trades)
        
        if total_trades == 0:
            logger.warning(f"\n{'='*80}\n  No trades executed for {symbol}\n{'='*80}")
            return

        # Calculate basic statistics
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        pnl_values = [t['pnl'] for t in trades]
        total_pnl = sum(pnl_values)
        avg_win = sum([pnl for pnl in pnl_values if pnl > 0]) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum([pnl for pnl in pnl_values if pnl <= 0]) / losing_trades if losing_trades > 0 else 0
        
        # Header
        logger.info(f"\n╔{'═'*78}╗")
        logger.info(f"║{f' BACKTEST SUMMARY FOR {symbol} ':^78}║")
        logger.info(f"╠{'═'*78}╣")
        
        # Configuration Section
        logger.info(f"║{' CONFIGURATION ':^78}║")
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║ {'Timeframes':<20}: {', '.join(self.config['timeframes']):<55} ║")
        logger.info(f"║ {'Period':<20}: {self.config['start_date']} to {self.config['end_date']:<33} ║")
        logger.info(f"║ {'Initial Balance':<20}: ${self.config['initial_balance']:<53,.2f} ║")
        
        # Performance Overview
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║{' PERFORMANCE OVERVIEW ':^78}║")
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║ {'Total Trades':<20}: {total_trades:<55d} ║")
        logger.info(f"║ {'Win Rate':<20}: {win_rate:<52.2f}% ║")
        logger.info(f"║ {'Total P&L':<20}: ${total_pnl:<53,.2f} ║")
        
        # Calculate profit factor safely
        profit_factor = abs(avg_win/avg_loss) if avg_loss != 0 else float('inf')
        profit_factor_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        logger.info(f"║ {'Profit Factor':<20}: {profit_factor_str:<55} ║")
        
        # Detailed Statistics
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║{' DETAILED STATISTICS ':^78}║")
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║ {'Winning Trades':<20}: {winning_trades:<55d} ║")
        logger.info(f"║ {'Losing Trades':<20}: {losing_trades:<55d} ║")
        logger.info(f"║ {'Average Win':<20}: ${avg_win:<53,.2f} ║")
        logger.info(f"║ {'Average Loss':<20}: ${abs(avg_loss):<53,.2f} ║")
        logger.info(f"║ {'Largest Win':<20}: ${max(pnl_values):<53,.2f} ║")
        logger.info(f"║ {'Largest Loss':<20}: ${min(pnl_values):<53,.2f} ║")
        
        max_dd = results.get('metrics', {}).get('max_drawdown', 0)
        logger.info(f"║ {'Max Drawdown':<20}: {max_dd*100:<52.2f}% ║")
        
        sharpe = results.get('metrics', {}).get('sharpe_ratio', 0)
        logger.info(f"║ {'Sharpe Ratio':<20}: {sharpe:<55.2f} ║")
        
        # Trade Analysis
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║{' TRADE ANALYSIS ':^78}║")
        logger.info(f"╟{'─'*78}╢")
        sl_hits = sum(1 for t in trades if t['exit_price'] == t['stop_loss'])
        tp_hits = sum(1 for t in trades if t['exit_price'] == t['take_profit'])
        other_closes = total_trades - sl_hits - tp_hits
        
        logger.info(f"║ {'Stop Loss Hits':<20}: {sl_hits:>4d} ({(sl_hits/total_trades)*100:>6.1f}%){' '*42} ║")
        logger.info(f"║ {'Take Profit Hits':<20}: {tp_hits:>4d} ({(tp_hits/total_trades)*100:>6.1f}%){' '*42} ║")
        logger.info(f"║ {'Other Closes':<20}: {other_closes:>4d} ({(other_closes/total_trades)*100:>6.1f}%){' '*42} ║")
        
        # Recent Trades
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║{' RECENT TRADES (Last 5) ':^78}║")
        logger.info(f"╟{'─'*78}╢")
        logger.info(f"║ {'Time':<19}{'Direction':<10}{'Entry':<11}{'Exit':<11}{'P&L':<10}{'Outcome':<15}║")
        logger.info(f"╟{'─'*78}╢")
        
        for trade in trades[-5:]:
            exit_time = pd.to_datetime(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time']
            outcome = "SL Hit" if trade['exit_price'] == trade['stop_loss'] else "TP Hit" if trade['exit_price'] == trade['take_profit'] else "Closed"
            logger.info(
                f"║ {exit_time.strftime('%Y-%m-%d %H:%M'):<19}"
                f"{trade['direction']:<10}"
                f"{trade['entry_price']:10.5f}"
                f"{trade['exit_price']:11.5f}"
                f"${trade['pnl']:8.2f}"
                f"{outcome:<15}║"
            )
        
        logger.info(f"╚{'═'*78}╝\n")

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

    def _save_symbol_results(self, results: Dict, symbol: str):
        """Save backtest results for a single symbol."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timeframe_str = '_'.join(self.config["timeframes"])
        date_range = f"{self.config['start_date'].replace('-', '')}_{self.config['end_date'].replace('-', '')}"
        file_prefix = f"backtest_{symbol}_{timeframe_str}_{date_range}"
        
        # Save JSON results
        json_filepath = f"{self.config['results_dir']}/{file_prefix}.json"
        with open(json_filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Save Excel results
        excel_filepath = f"{self.config['results_dir']}/{file_prefix}.xlsx"
        self._save_excel_results(results, excel_filepath, symbol)
        
    def _visualize_symbol_results(self, results: Dict, symbol: str):
        """Create visualization for a single symbol."""
        try:
            if not results.get('equity_curve'):
                logger.warning(f"No equity curve data available for {symbol}, skipping visualization")
                return
                
            # Handle both dictionary and list formats for equity curve
            if isinstance(results['equity_curve'][0], dict):
                dates = [e['timestamp'] for e in results['equity_curve']]
                equity = [e['balance'] for e in results['equity_curve']]
            else:
                dates = [e[0] for e in results['equity_curve']]
                equity = [e[1] for e in results['equity_curve']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name=f'{symbol} Equity Curve'
            ))
            
            fig.update_layout(
                title=f'Backtest Results - {symbol} Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity',
                template='plotly_dark'
            )
            
            timeframe_str = '_'.join(self.config["timeframes"])
            date_range = f"{self.config['start_date'].replace('-', '')}_{self.config['end_date'].replace('-', '')}"
            file_prefix = f"equity_curve_{symbol}_{timeframe_str}_{date_range}"
            fig.write_html(f"{self.config['results_dir']}/{file_prefix}.html")
            logger.info(f"Created equity curve visualization for {symbol}")
            
        except Exception as e:
            logger.error(f"Error creating visualization for {symbol}: {str(e)}")
            # Continue execution despite visualization error

    def _save_excel_results(self, results: Dict, excel_filepath: str, symbol: str):
        """Save backtest results to Excel file with multiple sheets."""
        try:
            import xlsxwriter
            
            # Create Excel workbook with custom formats
            workbook = xlsxwriter.Workbook(excel_filepath)
            
            # Add custom formats
            num_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            price_format = workbook.add_format({'num_format': '0.00000'})
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D9D9D9',
                'border': 1
            })
            
            # Trade List Sheet
            trades_df = pd.DataFrame([{
                'ID': t['id'],
                'Entry Time': t['entry_time'],
                'Exit Time': t['exit_time'],
                'Type': t['direction'],
                'Entry Price': t['entry_price'],
                'Exit Price': t['exit_price'],
                'Stop Loss': t['stop_loss'],
                'Take Profit': t['take_profit'],
                'Position Size': t['position_size'],
                'Risk Amount': t['risk_amount'],
                'PnL': t['pnl'],
                'R Multiple': t.get('r_multiple', 0),
                'Exit Reason': t.get('exit_reason', ''),
                'Holding Time (Hours)': t.get('holding_time_hours', 0),
                'MAE': t.get('max_adverse_excursion', 0),
                'MFE': t.get('max_favorable_excursion', 0)
            } for t in results['trades']])
            
            trades_sheet = workbook.add_worksheet('Trades')
            self._write_dataframe_to_sheet(trades_df, trades_sheet, header_format, num_format, percent_format, price_format)
            
            # Performance Metrics Sheet
            metrics_sheet = workbook.add_worksheet('Performance Metrics')
            metrics = results['metrics']
            
            # Write metrics to sheet
            metrics_sheet.write(0, 0, 'Metric')
            metrics_sheet.write(0, 1, 'Value')
            
            # Write overall metrics
            metrics_data = [
                ['Total Trades', metrics['total_trades']],
                ['Winning Trades', metrics['winning_trades']],
                ['Losing Trades', metrics['losing_trades']],
                ['Win Rate', metrics['win_rate']],
                ['Profit Factor', metrics['profit_factor']],
                ['Total Profit', metrics['total_profit']],
                ['Total Loss', metrics['total_loss']],
                ['Average Win', metrics['average_win']],
                ['Average Loss', metrics['average_loss']],
                ['Largest Win', metrics['largest_win']],
                ['Largest Loss', metrics['largest_loss']],
                ['Max Drawdown', metrics['max_drawdown']],
                ['Sharpe Ratio', metrics['sharpe_ratio']],
                ['Average Holding Time', metrics['average_holding_time']],
                ['Average MAE', metrics['average_mae']],
                ['Average MFE', metrics['average_mfe']],
                ['Risk Adjusted Return', metrics['risk_adjusted_return']],
            ]

            for i, (metric, value) in enumerate(metrics_data, 1):
                metrics_sheet.write(i, 0, metric)
                if 'Rate' in metric or 'Ratio' in metric or 'Drawdown' in metric:
                    metrics_sheet.write(i, 1, value, percent_format)
                elif any(term in metric for term in ['Profit', 'Loss', 'Win', 'MAE', 'MFE']):
                    metrics_sheet.write(i, 1, value, num_format)
                else:
                    metrics_sheet.write(i, 1, value)
            
            
            # Session Performance Sheet
            session_df = pd.DataFrame([{
                'Session': session,
                'Total Trades': data['trades'],
                'Wins': data['wins'],
                'Losses': data['losses'],
                'Win Rate': data['wins'] / data['trades'] if data['trades'] > 0 else 0,
                'Total PnL': data['total_pnl']
            } for session, data in metrics['session_performance'].items()])
            
            session_sheet = workbook.add_worksheet('Session Performance')
            self._write_dataframe_to_sheet(session_df, session_sheet, header_format, num_format, percent_format, price_format)
            
            # Market Conditions Sheet
            market_df = pd.DataFrame([{
                'Market Condition': condition,
                'Total Trades': data['trades'],
                'Wins': data['wins'],
                'Losses': data['losses'],
                'Win Rate': data['wins'] / data['trades'] if data['trades'] > 0 else 0,
                'Total PnL': data['total_pnl']
            } for condition, data in metrics['market_condition_performance'].items()])
            
            market_sheet = workbook.add_worksheet('Market Conditions')
            self._write_dataframe_to_sheet(market_df, market_sheet, header_format, num_format, percent_format, price_format)
            
            # Monthly Performance Sheet
            monthly_df = pd.DataFrame([{
                'Month': month,
                'Total Trades': data['trades'],
                'Wins': data['wins'],
                'Losses': data['losses'],
                'Win Rate': data['wins'] / data['trades'] if data['trades'] > 0 else 0,
                'Total PnL': data['total_pnl']
            } for month, data in metrics['monthly_performance'].items()])
            
            monthly_sheet = workbook.add_worksheet('Monthly Performance')
            self._write_dataframe_to_sheet(monthly_df, monthly_sheet, header_format, num_format, percent_format, price_format)
            
            workbook.close()
            logger.info(f"Excel results saved to: {excel_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Excel results: {str(e)}")
            
    def _write_dataframe_to_sheet(self, df: pd.DataFrame, worksheet, header_format, num_format, percent_format, price_format):
        """Helper method to write a DataFrame to an Excel worksheet with formatting."""
        # Write headers
        for col, header in enumerate(df.columns):
            worksheet.write(0, col, header, header_format)
        
        # Write data with appropriate formatting
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
        
        # Apply autofilter
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1) 

def print_detailed_analysis(results, config):
    """Print detailed analysis of backtest results."""
    logger.info("\n=== DETAILED BACKTEST ANALYSIS ===")
    
    try:
        if not results:
            logger.warning("No results to analyze")
            return
        
        for symbol, symbol_results in results.items():
            trades = symbol_results.get('trades', [])
            if not trades:
                logger.warning(f"No trades found for {symbol}, skipping analysis")
                continue
            
            logger.info(f"\n{'='*40} Analysis for {symbol} {'='*40}")
            
            # Basic Statistics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # Calculate metrics
            total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            
            # Print Overall Metrics
            logger.info("\nOverall Performance:")
            logger.info(f"{'Total Trades:':<20} {total_trades}")
            logger.info(f"{'Winning Trades:':<20} {len(winning_trades)}")
            logger.info(f"{'Losing Trades:':<20} {len(losing_trades)}")
            logger.info(f"{'Win Rate:':<20} {win_rate:.2%}")
            logger.info(f"{'Profit Factor:':<20} {profit_factor:.2f}")
            logger.info(f"{'Total Profit:':<20} ${total_profit:,.2f}")
            logger.info(f"{'Total Loss:':<20} ${total_loss:,.2f}")
            logger.info(f"{'Net Profit:':<20} ${total_profit - total_loss:,.2f}")
            
            # Detailed Trade Analysis Table
            logger.info("\n╔═══════════════════════════════════════ DETAILED TRADE ANALYSIS ═══════════════════════════════════════╗")
            logger.info("║ Trade # │ Time           │ Type │ Entry    │ SL       │ TP       │ Exit     │ R:R  │ Result │   P/L   ║")
            logger.info("╟─────────┼────────────────┼──────┼──────────┼──────────┼──────────┼──────────┼──────┼────────┼─────────║")
            
            for i, trade in enumerate(trades, 1):
                entry_time = pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')
                trade_type = trade['direction'][:4].upper()  # BUY or SELL
                entry = f"{trade['entry_price']:.5f}"
                sl = f"{trade['stop_loss']:.5f}"
                tp = f"{trade['take_profit']:.5f}"
                exit_price = f"{trade['exit_price']:.5f}"
                
                # Calculate R:R
                risk = abs(float(trade['entry_price']) - float(trade['stop_loss']))
                reward = abs(float(trade['take_profit']) - float(trade['entry_price']))
                rr = f"{reward/risk:.1f}" if risk != 0 else "N/A"
                
                # Determine result
                if trade['exit_price'] == trade['stop_loss']:
                    result = "SL Hit"
                elif trade['exit_price'] == trade['take_profit']:
                    result = "TP Hit"
                else:
                    result = "Closed"
                
                pnl = f"${trade['pnl']:,.2f}"
                
                logger.info(f"║ {i:7d} │ {entry_time:12s} │ {trade_type:4s} │ {entry:8s} │ {sl:8s} │ {tp:8s} │ {exit_price:8s} │ {rr:4s} │ {result:6s} │ {pnl:7s} ║")
            
            logger.info("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
            
            # Session Performance
            if 'metrics' in symbol_results and 'session_performance' in symbol_results['metrics']:
                logger.info("\nSession Performance:")
                for session, data in symbol_results['metrics']['session_performance'].items():
                    win_rate = data['wins'] / data['trades'] if data['trades'] > 0 else 0
                    logger.info(f"Session: {session:<10} Trades: {data['trades']:>3} Win Rate: {win_rate:>7.2%} P&L: ${data['total_pnl']:>10,.2f}")
            
            # Market Conditions
            if 'metrics' in symbol_results and 'market_condition_performance' in symbol_results['metrics']:
                logger.info("\nMarket Conditions Performance:")
                for condition, data in symbol_results['metrics']['market_condition_performance'].items():
                    win_rate = data['wins'] / data['trades'] if data['trades'] > 0 else 0
                    logger.info(f"Condition: {condition:<15} Trades: {data['trades']:>3} Win Rate: {win_rate:>7.2%} P&L: ${data['total_pnl']:>10,.2f}")
            
            # Monthly Performance
            if 'metrics' in symbol_results and 'monthly_performance' in symbol_results['metrics']:
                logger.info("\nMonthly Performance:")
                for month, data in sorted(symbol_results['metrics']['monthly_performance'].items()):
                    win_rate = data['wins'] / data['trades'] if data['trades'] > 0 else 0
                    logger.info(f"Month: {month:<10} Trades: {data['trades']:>3} Win Rate: {win_rate:>7.2%} P&L: ${data['total_pnl']:>10,.2f}")
            
            logger.info("\n" + "="*100 + "\n")
            
    except Exception as e:
        logger.error(f"Error in detailed analysis: {str(e)}")
        logger.exception("Detailed error:") 