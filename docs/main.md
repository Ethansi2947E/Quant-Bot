# Documentation for `main.py`

This script is the main entry point for the entire trading bot application. Its primary responsibilities are to initialize the environment, set up logging, instantiate and run the `TradingBot`, and handle graceful shutdown.

## Execution Flow

1.  **Environment Setup**:
    -   It first sets `PYTHONDONTWRITEBYTECODE` to `'1'` to prevent Python from creating `.pyc` cache files, keeping the project directory clean.
    -   It calls `load_dotenv(override=True)` to load environment variables from a `.env` file. This allows for easy configuration of sensitive data like API keys and account credentials without hardcoding them.

2.  **Configuration Loading**:
    -   After setting up the environment, it imports the necessary configuration dictionaries (`MT5_CONFIG`, `TRADING_CONFIG`, `TELEGRAM_CONFIG`, `LOG_CONFIG`) from `config.config`.

3.  **Logging Initialization**:
    -   It immediately calls `setup_logging(LOG_CONFIG)` from `src.utils.logging_setup`. This is a critical step that configures the Loguru-based logging system for the entire application. All subsequent log messages will adhere to the format and levels defined in `LOG_CONFIG`.

4.  **TradingBot Instantiation and Execution**:
    -   The `async def main()` function serves as the core asynchronous block.
    -   It bundles the imported configurations into a single `config` dictionary.
    -   It creates an instance of the `TradingBot`, passing the `config` object to its constructor.
    -   It calls `await trading_bot.start()`, which initializes all bot components (MT5 connection, strategies, etc.) and starts the main trading loop. The `start` method returns an `asyncio.Future` that signals when the bot has completed its shutdown process.
    -   The script then `await`s this `shutdown_future`, effectively pausing execution here until the bot is instructed to stop (e.g., via a Telegram command or `KeyboardInterrupt`).

5.  **Graceful Shutdown and Error Handling**:
    -   The entire bot's lifecycle is wrapped in a `try...finally` block.
    -   If any exception occurs during the bot's operation, it is caught, logged, and the `finally` block is executed.
    -   The `finally` block ensures that `await trading_bot.stop()` is always called, allowing the bot to close open positions (if configured), disconnect from services, and clean up resources properly.
    -   It also includes `except` blocks for `asyncio.CancelledError` and a general `Exception` to log specific shutdown scenarios.
    -   The `if __name__ == "__main__":` block ensures the script runs the `asyncio.run(main())` loop only when executed directly. It catches `KeyboardInterrupt` (Ctrl+C) to allow for a clean, user-initiated shutdown.

## How to Run

To start the trading bot, execute this file from the project's root directory:

```bash
python main.py
```
