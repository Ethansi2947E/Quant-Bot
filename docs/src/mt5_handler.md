# Documentation for `src/mt5_handler.py`

This file defines the `MT5Handler` class, which serves as a dedicated, low-level interface to the MetaTrader 5 trading terminal. It encapsulates all direct interactions with the `MetaTrader5` library, providing a clean, robust, and reusable API for the rest of the application.

## Core Responsibilities

1.  **Connection Management**: Handles initializing, logging into, and shutting down the connection to the MT5 terminal. It includes automatic reconnection logic to handle network interruptions gracefully.
2.  **Data Retrieval**: Provides methods to fetch various types of data from the broker's server, including:
    -   Historical and real-time market data (candlesticks/bars).
    -   Live tick data.
    -   Account information (balance, equity, margin).
    -   Symbol information (contract size, tick size, etc.).
    -   Open positions and historical orders/deals.
3.  **Order Execution**: Manages the lifecycle of trades by sending, modifying, and closing orders.
4.  **Error Handling and Resilience**: Implements retry logic, connection checks, and detailed error parsing to make interactions with the MT5 terminal more reliable.

## Key Features and Methods

### Singleton Pattern

-   **`get_instance()`**: The `MT5Handler` is implemented as a singleton. This ensures that only one instance of the handler exists throughout the application, preventing multiple, conflicting connections to the MT5 terminal.

### Connection and Account

-   **`initialize()`**: The primary method for establishing a connection. It handles shutdown of any previous connections, initialization, and login using credentials from `MT5_CONFIG`.
-   **`is_connected()`**: A robust check to verify that the connection to the terminal is active.
-   **`get_account_info()`**: Retrieves and returns a dictionary of key account metrics.

### Market Data

-   **`get_market_data(...)`**: Fetches historical OHLC (Open, High, Low, Close) data for a given symbol and timeframe, returning it as a Pandas DataFrame. It includes retry logic for connection issues.
-   **`get_last_tick(...)`**: Retrieves the most recent bid/ask price for a symbol.
-   **`get_latest_candle_time(...)`**: A lightweight method to get the timestamp of the most recent bar, used by the `TradingBot` to detect new candle events.

### Trade Execution

-   **`place_market_order(...)`**: The main function for executing market orders (buy or sell). This method is highly sophisticated and includes several critical pre-flight checks:
    -   **SL/TP Validation**: Adjusts stop-loss and take-profit levels if they are too close to the current market price, preventing broker rejections.
    -   **Volume Adjustment**: Calls `_adjust_volume_to_broker_limits` to ensure the trade volume complies with the symbol's minimum/maximum lot size and step size.
    -   **Retry Logic**: If an order fails due to a requote or other retry-able error, it will attempt to place the order again with adjusted parameters (e.g., increased slippage tolerance).
    -   **Filling Mode Handling**: It intelligently tries different order filling modes (e.g., `IOC`, `FOK`) if the default one is not supported by the broker for that symbol.
-   **`place_limit_order(...)`**: Places pending orders (buy/sell limit).
-   **`close_position(...)`**: Closes an open position by its ticket number.
-   **`modify_position(...)`**: Modifies the stop-loss and take-profit of an existing open position.

### Utility and Helper Methods

-   **`get_symbol_info(...)`**: Retrieves detailed contract specifications for a symbol.
-   **`_adjust_volume_to_broker_limits(...)`**: A crucial helper that ensures trade sizes are valid according to broker rules.
-   **`get_error_info(...)`**: Translates cryptic MT5 error codes into human-readable strings, which is invaluable for debugging.

## How It's Used

The `MT5Handler` is instantiated once (usually within the `TradingBot`) and then passed by reference to other components like `RiskManager`, `DataManager`, and `SignalProcessor` that require direct access to the broker's server. This centralized approach ensures consistent and stable communication with the MT5 terminal.
