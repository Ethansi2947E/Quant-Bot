from src.mt5_handler import MT5Handler
from src.utils.backtester import Backtester, select_data_loader, load_strategy
import asyncio
from datetime import datetime
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

# 1. Setup MT5 handler (singleton pattern)
mt5_handler = None  # Will be created only if needed

# 2. Choose symbol(s) and timeframe(s)
symbols = ["Volatility 10 Index",
        "Crash 500 Index",
        "Crash 1000 Index",
        "Boom 300 Index",
        "XAUUSD",
        "BTCUSD",
        "Boom 1000 Index",
        "Jump 50 Index",
        "Jump 75 Index",
        "Step Index",
        "Range Break 200 Index",]  # Use a symbol available in your MT5
timeframes = ['M15', 'H1']

# 3. Setup smart data loader (cache mode)
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 31)
data_loader = select_data_loader(
    'cache',
    directory='data/',         # Directory to store/load OHLCV CSVs
    mt5_handler=mt5_handler,
    num_bars=1000,              # Number of bars to fetch if not cached
    start_date=start_date,
    end_date=end_date
)

# 4. Load the strategy (BreakoutReversalStrategy)
strategy = load_strategy('BreakoutReversalStrategy', {
    'primary_timeframe': 'M15',
    'higher_timeframe': 'H1'
})

# 5. Create and run the backtester
backtester = Backtester(
    strategy=strategy,
    data_loader=data_loader,
    symbols=symbols,
    timeframes=timeframes,
    config={
        'initial_balance': 10000,
        'trade_log_path': 'exports/demo_trade_log.csv',
        'trade_log_json_path': 'exports/demo_trade_log.json',
        'results_json_path': 'exports/demo_results.json'
    }
)
asyncio.run(backtester.run())
backtester.report()