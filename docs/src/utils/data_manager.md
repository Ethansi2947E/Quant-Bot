# Documentation for `src/utils/data_manager.py`

This file defines the `DataManager` class, a utility responsible for handling all aspects of market data and trade data persistence. Its role is to act as an intermediary between the live data source (`MT5Handler`), the strategies that consume the data, and the database that stores historical trade records.

## Core Responsibilities

1.  **Data Caching**:
    -   Maintains an in-memory cache (`market_data_cache`) of market data (`pandas.DataFrame`s) for various symbols and timeframes.
    -   This caching mechanism is crucial for performance, as it prevents the bot from repeatedly requesting the same historical data from the MT5 terminal, which can be slow.

2.  **Data Requirements Registration**:
    -   Strategies register their data needs (symbol, timeframe, and required number of candles/lookback period) with the `DataManager` using the `register_timeframe` method.
    -   The `DataManager` uses this registry to ensure it fetches a sufficient amount of historical data for each strategy's indicators to initialize correctly.

3.  **Data Lifecycle Management**:
    -   **`update_data`**: The primary method for fetching the latest market data from `MT5Handler` and refreshing the cache. This is typically called by the `TradingBot`'s main loop when a new candle is detected.
    -   **`update_real_time_data`**: In a tick-based execution mode, this method can update the most recent (incomplete) candle in the cache with live tick data. This allows strategies to perform intra-bar analysis if needed.

4.  **Database Interaction (Persistence)**:
    -   **`init_db`**: Initializes a connection to a local SQLite database (`trading_bot.db`) and creates the necessary tables (`trades`, `signals`) if they don't already exist. It uses SQLAlchemy for ORM (Object-Relational Mapping).
    -   **`log_trade`**: Saves the details of a completed trade to the `trades` table in the database.
    -   **`log_signal`**: Saves generated trading signals to the `signals` table.
    -   **`synchronize_historical_trades`**: A powerful utility method that fetches the entire trade history from the MT5 terminal and "upserts" it into the local database. This ensures that the local database is a complete and accurate mirror of the broker's records.

## Key Features and Methods

-   **`__init__(self, mt5_handler)`**: Initializes the `DataManager`, takes an instance of `MT5Handler` to communicate with the MT5 terminal, and sets up the database connection.
-   **`get_market_data(symbol, timeframe)`**: The primary method for strategies to access the cached market data.
-   **`get_market_data_for_symbol(symbol, timeframes)`**: A convenience method that retrieves all the required DataFrames for a single symbol across multiple timeframes, which is exactly the format the `generate_signals` method expects.

## How It's Used

-   The `DataManager` is instantiated within the `TradingBot`.
-   During the bot's setup phase, each active strategy has its `required_timeframes` and `lookback_periods` registered with the `DataManager`.
-   The `TradingBot`'s main event loop is responsible for calling `update_data` whenever a new candle closes.
-   Strategies then call `get_market_data` or `get_market_data_for_symbol` to get the fresh data they need for analysis.
-   After a trade is executed and closed, the `TradingBot` or `SignalProcessor` calls `log_trade` to persist the record.
