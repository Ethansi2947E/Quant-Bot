'''
Unified Backtest Runner

This script is the main entry point for running backtests. It combines data
downloading and backtesting into a single, seamless process.

Usage:
    python backtest_runner.py --strategy <StrategyClassName> --symbol <SYMBOL> --timeframe <TIMEFRAME> --years <YEARS>

Example:
    python backtest_runner.py --strategy DanteBillsStrategy --symbol EURUSD --timeframe M15 --years 2
'''

import argparse
import asyncio
from pathlib import Path
import importlib.util
import re
from typing import Type
from datetime import datetime, timedelta

from loguru import logger

# Framework imports
from src.backtesting.engine import Backtester
from src.backtesting.data_loader import load_historical_data
from src.trading_bot import SignalGenerator
from src.mt5_handler import MT5Handler

def _download_data_if_needed(symbol: str, timeframe: str, years: float) -> Path:
    """
    Checks if data exists, and if not, downloads it from MT5.
    Returns the path to the data file.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / f"{symbol}_{timeframe}_{years}y.csv"

    if file_path.exists():
        logger.info(f"Data file found locally: {file_path}")
        return file_path

    logger.warning(f"Data file not found. Attempting to download from MT5...")
    logger.info(f"For fully repeatable tests, it's best to use pre-downloaded data.")

    mt5 = MT5Handler()
    if not mt5.initialize():
        raise ConnectionError("Failed to initialize MT5. Cannot download data.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    logger.info(f"Fetching data for {symbol} ({timeframe}) from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    data = mt5.get_historical_data(symbol, timeframe, start_date, end_date)
    mt5.shutdown()

    if data is None or data.empty:
        raise ValueError(f"Failed to retrieve data for {symbol}. No data was returned from MT5.")

    data.index.name = 'datetime'
    data.to_csv(file_path)
    logger.success(f"Successfully downloaded and saved {len(data)} records to {file_path}")
    return file_path

def _import_strategy_class(strategy_name: str) -> Type[SignalGenerator]:
    """Dynamically imports a strategy class from the src/strategy directory."""
    # Convert CamelCase to snake_case for the filename
    module_name = re.sub(r'(?!^)(?=[A-Z])', '_', strategy_name).lower().replace('_strategy', '')
    strategy_file = Path(f"src/strategy/{module_name}.py")
    
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy file not found for '{strategy_name}'. Expected at '{strategy_file}'")

    spec = importlib.util.spec_from_file_location(f"src.strategy.{module_name}", strategy_file)
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)

    StrategyClass = getattr(strategy_module, strategy_name, None)
    if StrategyClass and issubclass(StrategyClass, SignalGenerator):
        return StrategyClass
            
    raise TypeError(f"No class named '{strategy_name}' that inherits from SignalGenerator found in {strategy_file}")


async def main():
    """Main function to run the backtester."""
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest with integrated data fetching.")
    parser.add_argument("--strategy", required=True, help="The class name of the strategy to test (e.g., DanteBillsStrategy).")
    parser.add_argument("--symbol", required=True, help="The trading symbol (e.g., EURUSD, XAUUSD).")
    parser.add_argument("--timeframe", required=True, help="The timeframe (e.g., M1, M15, H1).")
    parser.add_argument("--years", type=float, default=1.0, help="The number of years of historical data to use.")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial account balance.")

    args = parser.parse_args()

    try:
        # 1. Get data (download if necessary)
        data_path = _download_data_if_needed(args.symbol, args.timeframe, args.years)
        data = load_historical_data(str(data_path), symbol=args.symbol)
        if data is None: return

        # 2. Load the strategy class
        StrategyClass = _import_strategy_class(args.strategy)

        # 3. Initialize and run the backtester
        backtester = Backtester(
            strategy_class=StrategyClass,
            data=data,
            symbol=args.symbol,
            initial_balance=args.balance,
            strategy_params={}
        )
        
        await backtester.run()

    except (FileNotFoundError, TypeError, ConnectionError, ValueError) as e:
        logger.error(e)
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    logger.add("backtest_logs.log", rotation="10 MB", level="DEBUG")
    asyncio.run(main())