from src.backtester import Backtester
from loguru import logger
import argparse
from datetime import datetime

def print_detailed_analysis(results, config):
    """Print detailed analysis of backtest results."""
    trades = results['trades']
    
    # Print configuration header
    logger.info("\n=== DETAILED BACKTEST ANALYSIS ===")
    logger.info(f"\nAnalysis for {', '.join(config['symbols'])} on {', '.join(config['timeframes'])} timeframe(s)")
    logger.info(f"Period: {config['start_date']} to {config['end_date']}")
    
    # Calculate additional metrics
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    # Profitability metrics
    total_profit = sum(t.pnl for t in winning_trades)
    total_loss = sum(t.pnl for t in losing_trades)
    avg_profit = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0
    largest_win = max((t.pnl for t in winning_trades), default=0)
    largest_loss = min((t.pnl for t in losing_trades), default=0)
    
    # Time-based analysis
    trades_by_hour = {}
    trades_by_day = {}
    for trade in trades:
        hour = trade.entry_time.hour
        day = trade.entry_time.strftime('%A')
        trades_by_hour[hour] = trades_by_hour.get(hour, 0) + 1
        trades_by_day[day] = trades_by_day.get(day, 0) + 1
    
    # Print detailed results
    logger.info("\n=== DETAILED BACKTEST ANALYSIS ===")
    
    logger.info("\nProfitability Analysis:")
    logger.info(f"| {'Metric':<30} | {'Value':<20} |")
    logger.info(f"| {'-'*30} | {'-'*20} |")
    logger.info(f"| {'Total Winning Trades':<30} | {len(winning_trades):>20} |")
    logger.info(f"| {'Total Losing Trades':<30} | {len(losing_trades):>20} |")
    logger.info(f"| {'Average Profit per Winning Trade':<30} | ${avg_profit:>19.2f} |")
    logger.info(f"| {'Average Loss per Losing Trade':<30} | ${avg_loss:>19.2f} |")
    logger.info(f"| {'Largest Win':<30} | ${largest_win:>19.2f} |")
    logger.info(f"| {'Largest Loss':<30} | ${largest_loss:>19.2f} |")
    
    logger.info("\nTime Analysis:")
    logger.info("Trades by Hour:")
    logger.info(f"| {'Hour':<10} | {'Number of Trades':<20} |")
    logger.info(f"| {'-'*10} | {'-'*20} |")
    for hour in sorted(trades_by_hour.keys()):
        logger.info(f"| {hour:02d}:00 - {trades_by_hour[hour]:<20} |")
    
    logger.info("\nTrades by Day:")
    logger.info(f"| {'Day':<10} | {'Number of Trades':<20} |")
    logger.info(f"| {'-'*10} | {'-'*20} |")
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        logger.info(f"| {day:<10} | {trades_by_day.get(day, 0):<20} |")
    
    logger.info("\nConsecutive Analysis:")
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for trade in trades:
        if trade.pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    logger.info(f"| {'Max Consecutive Wins':<30} | {max_consecutive_wins:<20} |")
    logger.info(f"| {'Max Consecutive Losses':<30} | {max_consecutive_losses:<20} |")
    
    logger.info("\n=== END OF DETAILED ANALYSIS ===\n")

def main():
    """Run backtest with command line arguments."""
    parser = argparse.ArgumentParser(description='Run trading bot backtest')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-03-14', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD', 'GBPUSD'], help='Symbols to trade')
    parser.add_argument('--timeframes', nargs='+', default=['H1', 'H4'], help='Timeframes to analyze')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (1% = 0.01)')
    
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
        "results_dir": "backtest_results"
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
    logger.info("=" * 50 + "\n")
    
    try:
        backtester = Backtester(config)
        results = backtester.run_backtest()
        
        # Calculate overall metrics
        total_trades = len(results['trades'])
        total_pnl = sum(t.pnl for t in results['trades'])
        winning_trades = len([t for t in results['trades'] if t.pnl > 0])
        logger.info("\nBacktest Results Summary:")
        logger.info("==================================================")
        logger.info(f"| {'Metric':<20} | {'Value':<20} |")
        logger.info("--------------------------------------------------")
        logger.info(f"| {'Total Trades':<20} | {total_trades:>20} |")
        logger.info(f"| {'Total PnL':<20} | ${total_pnl:>19.2f} |")
        logger.info(f"| {'Win Rate':<20} | {(winning_trades/total_trades*100):>19.2f}% |" if total_trades > 0 else "| Win Rate              | N/A                  |")
        logger.info(f"| {'Final Balance':<20} | ${(config['initial_balance'] + total_pnl):>19.2f} |")
        logger.info("==================================================\n")
        
        # Print detailed analysis
        print_detailed_analysis(results, config)
        
    except Exception as e:
        logger.error(f"Error during backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    main() 