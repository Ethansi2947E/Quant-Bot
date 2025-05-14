# === Backtest Demo ===
#
# To change the strategy, set STRATEGY_NAME to the class name (e.g. 'PriceActionSRStrategy', 'BreakoutReversalStrategy', 'ConfluencePriceActionStrategy')
# Set PRIMARY_TIMEFRAME (and HIGHER_TIMEFRAME if the strategy requires two timeframes).
# You do NOT need to specify other parameters—they will use the strategy's own defaults.
#
# Example:
#   STRATEGY_NAME = 'BreakoutReversalStrategy'
#   PRIMARY_TIMEFRAME = 'M15'
#   HIGHER_TIMEFRAME = 'H1'  # Only if needed
#
# The script will inspect the strategy's __init__ and only pass the timeframes and required params.

# Prevent __pycache__ creation (no .pyc files)
import sys
sys.dont_write_bytecode = True

from src.mt5_handler import MT5Handler
from src.backtest.backtester import Backtester, select_data_loader
import asyncio
from datetime import datetime
from loguru import logger
import re
import sys
import importlib
import inspect
from pathlib import Path
from dotenv import load_dotenv

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

# === STRATEGY SELECTION ===
STRATEGY_NAME = 'PriceActionSRStrategy'  # e.g. 'PriceActionSRStrategy', 'BreakoutReversalStrategy', 'ConfluencePriceActionStrategy'
PRIMARY_TIMEFRAME = 'M15'
HIGHER_TIMEFRAME = 'H1'  # Only used if strategy requires it

# === EXPORT DIRECTORY STRUCTURE ===
# Results will be saved in: src/backtest/results/{year}/{symbol_or_multi_symbol}/{strategy}/{period}/
# Example: src/backtest/results/2025/XAUUSD/ConfluencePriceActionStrategy/2025-01-01_to_2025-01-31/

# 1. Setup MT5 handler (singleton pattern)
mt5_handler = None  # Will be created only if needed

# 2. Choose symbol(s) and timeframe(s)
symbols = [
        #"Volatility 10 Index",
        #"Crash 500 Index",
        # "Crash 1000 Index",
        # "Boom 300 Index",
        "XAUUSD",
        #"BTCUSD",
        # "Boom 1000 Index",
        #"Jump 50 Index",
         #"Jump 75 Index",
        #"Step Index",
        #"Range Break 200 Index",
        ]  # Use a symbol available in your MT5
timeframes = [PRIMARY_TIMEFRAME]
if HIGHER_TIMEFRAME and HIGHER_TIMEFRAME != PRIMARY_TIMEFRAME:
    timeframes.append(HIGHER_TIMEFRAME)

# 3. Setup smart data loader (cache mode)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
data_loader = select_data_loader(
    'cache',
    directory='src/backtest/data',         # Directory to store/load OHLCV CSVs (now structured by year/symbol/timeframe)
    mt5_handler=mt5_handler,
    num_bars=1000,              # Number of bars to fetch if not cached
    start_date=start_date,
    end_date=end_date
)

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# 4. Dynamically import and instantiate the selected strategy
module_name = f"src.strategy.{camel_to_snake(STRATEGY_NAME)}"
strategy_module = importlib.import_module(module_name)
strategy_class = getattr(strategy_module, STRATEGY_NAME)

# Inspect the __init__ signature to determine which timeframes to pass
sig = inspect.signature(strategy_class.__init__)
params = sig.parameters
init_kwargs = {}
if 'primary_timeframe' in params:
    init_kwargs['primary_timeframe'] = PRIMARY_TIMEFRAME
if 'higher_timeframe' in params and HIGHER_TIMEFRAME and HIGHER_TIMEFRAME != PRIMARY_TIMEFRAME:
    init_kwargs['higher_timeframe'] = HIGHER_TIMEFRAME
# All other params will use the strategy's defaults
strategy = strategy_class(**init_kwargs)

# 0. Compute export directory for this run
period_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
year_str = str(start_date.year)
symbol_folder = symbols[0] if len(symbols) == 1 else "multi-symbol"
strategy_folder = STRATEGY_NAME
EXPORT_DIR = Path(f"src/backtest/results/{year_str}/{symbol_folder}/{strategy_folder}/{period_str}")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# 5. Create and run the backtester
backtester = Backtester(
    strategy=strategy,
    data_loader=data_loader,
    symbols=symbols,
    timeframes=timeframes,
    config={
        'initial_balance': 1000,
        'trade_log_path': str(EXPORT_DIR / 'trade_log.csv'),
        'trade_log_json_path': str(EXPORT_DIR / 'trade_log.json'),
        'results_json_path': str(EXPORT_DIR / 'results.json'),
        'profile': False
    },
    export_config={
        'csv': str(EXPORT_DIR / 'trade_log.csv'),
        'json': str(EXPORT_DIR / 'trade_log.json'),
        'excel': str(EXPORT_DIR / 'trade_log.xlsx'),
        'markdown': str(EXPORT_DIR / 'report.md'),
        'html': str(EXPORT_DIR / 'report.html'),
        'report_json': str(EXPORT_DIR / 'report.json')
    }
)
asyncio.run(backtester.run())
backtester.report()

# Load environment variables
load_dotenv(override=True)