from src.backtester import Backtester
from loguru import logger
import argparse
import pandas as pd

def print_detailed_analysis(results, config):
    """Print detailed analysis of backtest results."""
    logger.info("\n=== DETAILED BACKTEST ANALYSIS ===")
    # Debug: Log the structure of results
    logger.debug(f"Results received: {list(results.keys())}")
    
    try:
        if not results:
            logger.warning("No results to analyze")
            return
        
        for symbol, symbol_results in results.items():
            # Debug: Log keys of symbol_results
            logger.debug(f"Processing symbol {symbol} with keys: {list(symbol_results.keys())}")
            trades = symbol_results.get('trades', [])
            if not trades:
                logger.warning(f"No trades found for {symbol}, skipping analysis")
                continue
            
            total_trades = len(trades)
            if total_trades == 0:
                logger.warning("No trades found in results")
                return
                
            # Calculate profit metrics consistently
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            break_even_trades = [t for t in trades if t['pnl'] == 0]
            
            total_profit = sum(t['pnl'] for t in winning_trades)
            total_loss = abs(sum(t['pnl'] for t in losing_trades))  # Make sure loss is positive
            
            # Calculate profit factor properly
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            
            # Calculate R multiples
            for trade in trades:
                if trade['stop_loss'] and trade['entry_price']:
                    risk = abs(trade['entry_price'] - trade['stop_loss'])
                    if risk != 0:
                        trade['r_multiple'] = trade['pnl'] / (risk * 100000)  # Convert pips to dollars
                    else:
                        trade['r_multiple'] = 0
                else:
                    trade['r_multiple'] = 0
            
            # Calculate MAE and MFE properly
            for trade in trades:
                mae = trade.get('max_adverse_excursion', 0)
                mfe = trade.get('max_favorable_excursion', 0)
                if trade['entry_price']:
                    # Convert to percentage of entry price for normalization
                    trade['mae_pct'] = (mae / trade['entry_price']) * 100 if mae else 0
                    trade['mfe_pct'] = (mfe / trade['entry_price']) * 100 if mfe else 0
                else:
                    trade['mae_pct'] = 0
                    trade['mfe_pct'] = 0
            
            # Calculate averages
            avg_r_multiple = sum(t.get('r_multiple', 0) for t in trades) / len(trades) if trades else 0
            avg_mae = sum(t.get('mae_pct', 0) for t in trades) / len(trades) if trades else 0
            avg_mfe = sum(t.get('mfe_pct', 0) for t in trades) / len(trades) if trades else 0
            
            # Print Risk Analysis with corrected metrics
            logger.info("\nRisk Analysis:")
            logger.info(f"| {'Metric':<30} | {'Value':<20} |")
            logger.info(f"| {'-'*30} | {'-'*20} |")
            logger.info(f"| {'Profit Factor':<30} | {profit_factor:>19.2f} |")
            logger.info(f"| {'Average R Multiple':<30} | {avg_r_multiple:>19.2f} |")
            logger.info(f"| {'Average MAE (%)':<30} | {avg_mae:>19.2f} |")
            logger.info(f"| {'Average MFE (%)':<30} | {avg_mfe:>19.2f} |")
            logger.info(f"| {'Max MAE (%)':<30} | {max((t.get('mae_pct', 0) for t in trades), default=0):>19.2f} |")
            logger.info(f"| {'Max MFE (%)':<30} | {max((t.get('mfe_pct', 0) for t in trades), default=0):>19.2f} |")
            
            # Print individual trade details with corrected metrics
            logger.info("\nDetailed Trade List:")
            logger.info(f"| {'ID':<4} | {'Dir':<6} | {'Entry Time':<19} | {'Exit Time':<19} | {'Entry':<10} | {'Exit':<10} | {'PnL':<10} | {'R':<6} | {'MAE%':<6} | {'MFE%':<6} | {'Reason':<15} | {'Strategy':<15} |")
            logger.info(f"| {'-'*4} | {'-'*6} | {'-'*19} | {'-'*19} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*15} | {'-'*15} |")
            
            for trade in trades:
                logger.info(
                    f"| {trade['id']:<4} | "
                    f"{trade['direction']:<6} | "
                    f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'):<19} | "
                    f"{trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'):<19} | "
                    f"{trade['entry_price']:<10.5f} | "
                    f"{trade['exit_price']:<10.5f} | "
                    f"${trade['pnl']:<9.2f} | "
                    f"{trade.get('r_multiple', 0):<6.2f} | "
                    f"{trade.get('mae_pct', 0):<6.2f} | "
                    f"{trade.get('mfe_pct', 0):<6.2f} | "
                    f"{trade.get('exit_reason', 'Unknown'):<15} | "
                    f"{trade.get('strategy', 'Unknown'):<15} |"
                )
            
            # Time Analysis
            trades_by_hour = {}
            trades_by_day = {}
            trades_by_session = {}
            
            for trade in trades:
                if isinstance(trade['entry_time'], str):
                    entry_time = pd.to_datetime(trade['entry_time'])
                else:
                    entry_time = trade['entry_time']
                    
                hour = entry_time.hour
                day = entry_time.strftime('%A')
                session = trade.get('market_conditions', {}).get('session', 'Unknown')
                
                trades_by_hour[hour] = trades_by_hour.get(hour, 0) + 1
                trades_by_day[day] = trades_by_day.get(day, 0) + 1
                trades_by_session[session] = trades_by_session.get(session, 0) + 1
        
            logger.info("\nTime Analysis:")
            logger.info("\nTrades by Hour:")
            for hour in sorted(trades_by_hour.keys()):
                logger.info(f"{hour:02d}:00 - {trades_by_hour[hour]} trades")
                
            logger.info("\nTrades by Day:")
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if day in trades_by_day:
                    logger.info(f"{day}: {trades_by_day[day]} trades")
                    
            logger.info("\nTrades by Session:")
            for session in sorted(trades_by_session.keys()):
                logger.info(f"{session}: {trades_by_session[session]} trades")
                
            # Strategy Analysis
            trades_by_strategy = {}
            for trade in trades:
                strategy = trade.get('strategy', 'Unknown')
                trades_by_strategy[strategy] = trades_by_strategy.get(strategy, 0) + 1
            
            logger.info("\nStrategy Analysis:")
            logger.info(f"| {'Strategy':<30} | {'Count':<10} | {'Win Rate':<10} | {'Avg Profit':<10} |")
            logger.info(f"| {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} |")
            for strategy in sorted(trades_by_strategy.keys()):
                strategy_trades = [t for t in trades if t.get('strategy') == strategy]
                strategy_winners = len([t for t in strategy_trades if t['pnl'] > 0])
                strategy_win_rate = strategy_winners / len(strategy_trades) if strategy_trades else 0
                strategy_avg_profit = sum(t['pnl'] for t in strategy_trades) / len(strategy_trades) if strategy_trades else 0
                strategy_name = str(strategy) if strategy is not None else "Unknown"
                logger.info(f"| {strategy_name:<30} | {trades_by_strategy[strategy]:<10} | {strategy_win_rate*100:>8.1f}% | ${strategy_avg_profit:>9.2f} |")
                
            # Analyze each symbol separately
            for symbol, symbol_results in results.items():
                trades = symbol_results['trades']
                if not trades:
                    logger.info(f"\nNo trades found for {symbol}")
                    continue
                
                logger.info(f"\n=== Analysis for {symbol} ===")
                logger.info(f"Period: {config['start_date']} to {config['end_date']}")
                
                # Basic Statistics
                total_trades = len(trades)
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] < 0]
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                
                logger.info("\nBasic Statistics:")
                logger.info(f"| {'Metric':<30} | {'Value':<20} |")
                logger.info(f"| {'-'*30} | {'-'*20} |")
                logger.info(f"| {'Total Trades':<30} | {total_trades:>20} |")
                logger.info(f"| {'Winning Trades':<30} | {len(winning_trades):>20} |")
                logger.info(f"| {'Losing Trades':<30} | {len(losing_trades):>20} |")
                logger.info(f"| {'Win Rate':<30} | {win_rate*100:>19.2f}% |")
                
                # Profitability Analysis
                total_profit = sum(t['pnl'] for t in winning_trades)
                total_loss = sum(t['pnl'] for t in losing_trades)
                net_profit = total_profit + total_loss
                avg_win = total_profit / len(winning_trades) if winning_trades else 0
                avg_loss = total_loss / len(losing_trades) if losing_trades else 0
                largest_win = max((t['pnl'] for t in winning_trades), default=0)
                largest_loss = min((t['pnl'] for t in losing_trades), default=0)
            
                logger.info("\nProfitability Analysis:")
                logger.info(f"| {'Metric':<30} | {'Value':<20} |")
                logger.info(f"| {'-'*30} | {'-'*20} |")
                logger.info(f"| {'Total Profit':<30} | ${total_profit:>19.2f} |")
                logger.info(f"| {'Total Loss':<30} | ${total_loss:>19.2f} |")
                logger.info(f"| {'Net Profit':<30} | ${net_profit:>19.2f} |")
                logger.info(f"| {'Profit Factor':<30} | {profit_factor:>19.2f} |")
                logger.info(f"| {'Average Win':<30} | ${avg_win:>19.2f} |")
                logger.info(f"| {'Average Loss':<30} | ${avg_loss:>19.2f} |")
                logger.info(f"| {'Largest Win':<30} | ${largest_win:>19.2f} |")
                logger.info(f"| {'Largest Loss':<30} | ${largest_loss:>19.2f} |")
                
                # Market Conditions Analysis
                trades_by_trend = {}
                trades_by_volatility = {}
                
                for trade in trades:
                    market_conditions = trade.get('market_conditions', {})
                    trend = market_conditions.get('trend', 'Unknown')
                    volatility = market_conditions.get('volatility', 'Unknown')
                    
                    trades_by_trend[trend] = trades_by_trend.get(trend, 0) + 1
                    trades_by_volatility[volatility] = trades_by_volatility.get(volatility, 0) + 1
                
                logger.info("\nMarket Conditions Analysis:")
                logger.info("\nTrades by Trend:")
                logger.info(f"| {'Trend':<15} | {'Count':<10} | {'Win Rate':<10} | {'Avg Profit':<10} |")
                logger.info(f"| {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10} |")
                for trend in sorted(trades_by_trend.keys()):
                    trend_trades = [t for t in trades if t.get('market_conditions', {}).get('trend') == trend]
                    trend_winners = len([t for t in trend_trades if t['pnl'] > 0])
                    trend_win_rate = trend_winners / len(trend_trades) if trend_trades else 0
                    trend_avg_profit = sum(t['pnl'] for t in trend_trades) / len(trend_trades) if trend_trades else 0
                    trend_name = str(trend) if trend is not None else "Unknown"
                    logger.info(f"| {trend_name:<15} | {trades_by_trend[trend]:<10} | {trend_win_rate*100:>8.1f}% | ${trend_avg_profit:>9.2f} |")
                
                # Consecutive Analysis
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                current_wins = 0
                current_losses = 0
                
                for trade in trades:
                    if trade['pnl'] > 0:
                        current_wins += 1
                        current_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, current_wins)
                    else:
                        current_losses += 1
                        current_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, current_losses)
                
                logger.info("\nConsecutive Trade Analysis:")
                logger.info(f"| {'Metric':<30} | {'Value':<20} |")
                logger.info(f"| {'-'*30} | {'-'*20} |")
                logger.info(f"| {'Max Consecutive Wins':<30} | {max_consecutive_wins:>20} |")
                logger.info(f"| {'Max Consecutive Losses':<30} | {max_consecutive_losses:>20} |")
                
                logger.info("\n" + "="*80 + "\n")
    except Exception as e:
        logger.error(f"Error in detailed analysis: {str(e)}")

def main():
    """Run backtest with command line arguments."""
    parser = argparse.ArgumentParser(description='Run trading bot backtest')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-03-14', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD', 'GBPUSD'], help='Symbols to trade')
    parser.add_argument('--timeframes', nargs='+', default=['H1', 'H4'], help='Timeframes to analyze')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (1% = 0.01)')
    parser.add_argument('--signal-generator', type=str, default='default', 
                        choices=['default', 'signal_generator1'], 
                        help='Signal generator to use (default or signal_generator1)')
    parser.add_argument('--use-cached-data', action='store_true', 
                        help='Force using cached data instead of downloading from MT5')
    parser.add_argument('--download-data-only', action='store_true',
                        help='Download data without running backtest (to populate cache)')
    parser.add_argument('--no-fallback', action='store_true',
                        help='Disable fallback to basic signal generator when custom signal generator returns no signals')
    
    args = parser.parse_args()
    
    config = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": args.symbols,
        "timeframes": args.timeframes,
        "initial_balance": args.initial_balance,
        "risk_per_trade": args.risk_per_trade,
        "commission": 0.00007,  # 0.7 pips commission per trade
        "enable_visualization": True,
        "save_results": True,
        "results_dir": "backtest_results",
        "use_cached_data": args.use_cached_data,  # Add this parameter to control cached data usage
        "use_signal_fallback": not args.no_fallback  # Control fallback behavior
    }
    
    logger.info("=" * 50)
    logger.info("Starting backtest with configuration:")
    logger.info(f"| {'Parameter':<20} | {'Value':<20} |")
    logger.info(f"| {'-'*20} | {'-'*20} |")
    logger.info(f"| {'Start Date':<20} | {config['start_date']:<20} |")
    logger.info(f"| {'End Date':<20} | {config['end_date']:<20} |")
    logger.info(f"| {'Symbols':<20} | {', '.join(config['symbols']):<20} |")
    logger.info(f"| {'Timeframes':<20} | {', '.join(config['timeframes']):<20} |")
    logger.info(f"| {'Initial Balance':<20} | ${config['initial_balance']:.2f} |")
    logger.info(f"| {'Risk per Trade':<20} | {config['risk_per_trade']*100:.1f}% |")
    logger.info(f"| {'Signal Generator':<20} | {args.signal_generator:<20} |")
    logger.info(f"| {'Use Cached Data':<20} | {'Yes' if config['use_cached_data'] else 'No':<20} |")
    logger.info(f"| {'Use Signal Fallback':<20} | {'Yes' if config['use_signal_fallback'] else 'No':<20} |")
    logger.info(f"| {'Download Only':<20} | {'Yes' if args.download_data_only else 'No':<20} |")
    logger.info("=" * 50 + "\n")
    
    try:
        backtester = Backtester(config)
        
        # Special case: Download data only
        if args.download_data_only:
            logger.info("Download data only mode - downloading data for all symbols and timeframes to populate cache")
            for symbol in config['symbols']:
                for timeframe in config['timeframes']:
                    logger.info(f"Downloading data for {symbol} {timeframe}")
                    data = backtester.get_historical_data(symbol, timeframe, config['start_date'], config['end_date'])
                    if data is not None and not data.empty:
                        logger.info(f"Successfully downloaded {len(data)} candles for {symbol} {timeframe}")
                    else:
                        logger.warning(f"Failed to download data for {symbol} {timeframe}")
            logger.info("Data download completed")
            return
        
        # Normal backtest mode
        logger.info(f"Running backtest with signal generator: {args.signal_generator}")
        results = backtester.run_backtest_with_generator(args.signal_generator)
        
        if not results:
            logger.error("No results generated from backtest")
            return
            
        # Calculate overall metrics across all symbols
        all_trades = []
        for symbol_results in results.values():
            if symbol_results and 'trades' in symbol_results:
                all_trades.extend(symbol_results['trades'])
        
        if not all_trades:
            logger.warning("No trades were executed during the backtest period")
            return
            
        total_trades = len(all_trades)
        total_pnl = sum(t['pnl'] for t in all_trades)
        winning_trades = len([t for t in all_trades if t['pnl'] > 0])
        
        logger.info("\nOverall Backtest Results Summary:")
        logger.info("==================================================")
        logger.info(f"| {'Metric':<20} | {'Value':<20} |")
        logger.info("--------------------------------------------------")
        logger.info(f"| {'Signal Generator':<20} | {args.signal_generator:<20} |")
        logger.info(f"| {'Total Trades':<20} | {total_trades:>20} |")
        logger.info(f"| {'Total PnL':<20} | ${total_pnl:>19.2f} |")
        logger.info(f"| {'Win Rate':<20} | {(winning_trades/total_trades*100):>19.2f}% |" if total_trades > 0 else "| Win Rate              | N/A                  |")
        logger.info(f"| {'Final Balance':<20} | ${(config['initial_balance'] + total_pnl):>19.2f} |")
        logger.info("==================================================\n")
        
        # Analyze strategies used
        strategies = {}
        for trade in all_trades:
            strategy = trade.get('strategy', 'Unknown')
            if strategy not in strategies:
                strategies[strategy] = 0
            strategies[strategy] += 1
        
        logger.info("\nStrategies Used:")
        logger.info("==================================================")
        for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"| {strategy:<30} | {count:>10} trades |")
        logger.info("==================================================\n")
        
        # Print detailed analysis for each symbol
        print_detailed_analysis(results, config)
        
    except Exception as e:
        logger.error(f"Error during backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    main() 