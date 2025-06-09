import sys
sys.dont_write_bytecode = True

from src.backtest.backtester import Backtester, select_data_loader
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
STRATEGY_NAME = 'PriceActionSRStrategy'  # e.g. 'PriceActionSRStrategy', 'BreakoutReversalStrategy', 'ConfluencePriceActionStrategy', 'BreakoutTradingStrategy' (for backtesting)
PRIMARY_TIMEFRAME = 'M15'
HIGHER_TIMEFRAME = 'H1'  # Only used if strategy requires it

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

# 0. Compute export directory for this run (moved up)
period_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
year_str = str(start_date.year)
symbol_folder = symbols[0] if len(symbols) == 1 else "multi-symbol"
strategy_folder = STRATEGY_NAME
EXPORT_DIR = Path(f"src/backtest/results/{year_str}/{symbol_folder}/{strategy_folder}/{period_str}")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Config for data loading and backtesting
# Set allow_external_fetch to False for environments like Google Colab
backtest_run_config = {
    'initial_balance': 1000,
    'trade_log_path': str(EXPORT_DIR / 'trade_log.csv'), # Will be overridden by export_config in Backtester
    'trade_log_json_path': str(EXPORT_DIR / 'trade_log.json'), # Will be overridden by export_config
    'results_json_path': str(EXPORT_DIR / 'results.json'), # Will be overridden by export_config
    'profile': False,
    'allow_external_fetch': True  # <<< SET TO FALSE FOR COLAB/OFFLINE RUNS
}

data_loader = select_data_loader(
    'cache',
    directory='src/backtest/data',
    mt5_handler=mt5_handler, # mt5_handler instance, can be None
    num_bars=1000,
    start_date=start_date,
    end_date=end_date,
    config=backtest_run_config # Pass the config here
)

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# 4. Dynamically import and instantiate the selected strategy
module_name = f"src.strategy.{camel_to_snake(STRATEGY_NAME)}"
strategy_module = importlib.import_module(module_name)
strategy_class = getattr(strategy_module, STRATEGY_NAME)

# NOTE: If using BreakoutTradingStrategy, default settings now use a looser consolidation filter (bb_squeeze_factor=1.2, min_consolidation_bars=5),
# a more permissive volume filter (>=, 0.9x tolerance), and debugging aids (processed_bars clearing, wait_for_confirmation_candle=False).

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

# 5. Create and run the backtester
backtester = Backtester(
    strategy=strategy,
    data_loader=data_loader,
    symbols=symbols,
    timeframes=timeframes,
    config=backtest_run_config, # Pass the main config dict
    export_config={
        'csv': str(EXPORT_DIR / 'trade_log.csv'),
        'json': str(EXPORT_DIR / 'trade_log.json'),
        'excel': str(EXPORT_DIR / 'trade_log.xlsx'),
        'markdown': str(EXPORT_DIR / 'report.md'),
        'html': str(EXPORT_DIR / 'report.html'),
        'report_json': str(EXPORT_DIR / 'report.json')
    }
)
backtester.run()
backtester.report()

# Load environment variables
load_dotenv(override=True)