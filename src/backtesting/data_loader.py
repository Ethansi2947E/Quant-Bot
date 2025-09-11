'''
Backtesting Data Loader

This module provides a utility for loading historical market data from CSV files
for use in the backtesting engine.
'''

import pandas as pd
from loguru import logger
from typing import Optional


def load_historical_data(file_path: str, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Loads historical OHLCV data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The full path to the CSV file.
        symbol (Optional[str]): The symbol of the asset (for logging purposes).

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the historical data, or None if loading fails.
    """
    logger.info(f"Loading historical data for {symbol or 'unknown symbol'} from {file_path}...")

    try:
        data = pd.read_csv(
            file_path,
            parse_dates=['datetime'],
            index_col='datetime'
        )

        # --- Data Validation ---
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(
                f"Data validation failed. CSV must contain: {required_columns}. " 
                f"Found: {list(data.columns)}"
            )
            return None

        # Ensure data types are correct
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with any NaN values that might have been introduced
        data.dropna(inplace=True)

        if data.empty:
            logger.error("Data is empty after cleaning. Please check the source file.")
            return None

        logger.success(f"Successfully loaded {len(data)} candles for {symbol or 'asset'}.")
        return data

    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}: {e}")
        return None
