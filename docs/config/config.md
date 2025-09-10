# Documentation for `config/config.py`

This file centralizes all configuration parameters for the trading bot, making it easy to manage settings for different environments and strategies. It uses environment variables for sensitive information like API keys and credentials, loading them via `python-dotenv`.

## Key Components

### 1. Environment and Path Setup

-   **`load_dotenv()`**: Loads variables from a `.env` file into the environment. This is the standard way to handle secrets and environment-specific settings without hardcoding them.
-   **`BASE_DIR`**: Defines the project's root directory, allowing for robust and platform-independent file pathing.

### 2. Configuration Dictionaries

The file defines several dictionaries that group related settings:

#### `MT5_CONFIG`

Handles connection parameters for the MetaTrader 5 terminal.

-   `server`: The name of the broker's server.
-   `login`: The MT5 account number.
-   `password`: The MT5 account password.
-   `timeout`: Connection timeout in seconds.

*Note: These values are primarily loaded from environment variables for security (`MT5_SERVER`, `MT5_LOGIN`, `MT5_PASSWORD`).*

#### `TRADING_CONFIG`

This is the core configuration for the bot's trading behavior.

-   `account_type`: Can be `"demo"` or `"real"`.
-   `magic_number`: A unique integer that the bot attaches to its orders. This is crucial for identifying and managing its own trades, distinguishing them from manual trades or those from other bots.
-   `symbols`: A list of financial instruments the bot is allowed to trade.
-   `fixed_lot_size` & `use_fixed_lot_size`: If `use_fixed_lot_size` is `True`, the bot will use the `fixed_lot_size` for all trades, bypassing dynamic risk-based position sizing.
-   `max_lot_size`: The absolute maximum lot size the bot can use, regardless of calculation.
-   `max_daily_risk`: A deprecated or high-level risk parameter. More granular control is in `RISK_MANAGER_CONFIG`.
-   `allow_position_additions`: If `True`, the bot can add to existing positions (pyramiding).
-   `execution_mode`: Determines the bot's operational trigger.
    -   `'bar'`: The bot analyzes the market and generates signals only on the close of a new candle/bar. This is the standard, more stable approach.
    -   `'tick'`: The bot analyzes on every new incoming price tick. This is for high-frequency strategies and is more CPU-intensive.
-   `data_management`: A sub-dictionary for data handling settings.
    -   `use_direct_fetch`: If `True`, fetches fresh data before each analysis cycle.
    -   `validate_trades`: If `True`, validates signal price against current market price before execution.
-   `close_positions_on_shutdown`: If `True`, the bot will attempt to close all open positions when it shuts down.
-   `signal_generators`: A list of strategy class names (as strings) that the bot should load and run.

#### `TELEGRAM_CONFIG`

Settings for the Telegram notification and control bot.

-   `token`: The API token for the Telegram bot.
-   `allowed_users`: A list of Telegram user IDs who are authorized to interact with the bot.
-   `enabled`: A master switch to enable or disable the Telegram integration.

#### `LOG_CONFIG`

Defines the behavior of the logging system (powered by Loguru).

-   `use_file_logging`: If `True`, logs are written to a file. Can be controlled by the `LOG_TO_FILE` environment variable.
-   `log_file_path`: The path to the log file.
-   `level`: The minimum level of logs to record (e.g., `"INFO"`, `"DEBUG"`, `"TRACE"`).
-   `rotation`, `retention`, `compression`: Parameters for log file management (log rotation).
-   `format_console` & `format_file`: The format strings for console and file log messages, respectively.

#### `RISK_MANAGER_CONFIG`

Crucial settings that govern the `RiskManager`'s decisions.

-   `max_risk_per_trade`: The maximum percentage of account balance to risk on a single trade (e.g., `0.01` for 1%).
-   `max_drawdown`: The maximum allowable drawdown for the account before risk measures are taken.
-   `min_risk_reward_ratio`: The minimum required risk-to-reward ratio for a trade to be considered valid.
-   `max_daily_loss`: The maximum percentage of the account balance that can be lost in a single day before trading might be halted.
-   `max_concurrent_trades`: The maximum number of trades the bot can have open at one time.

#### `TRADE_EXIT_CONFIG`

Advanced configuration for managing trade exits, particularly for trailing stops and partial profit-taking.

-   `scalping`: Contains settings for a scalping-specific exit strategy, closing trades when a certain profit percentage is reached.
-   `trailing_stop`: A detailed sub-dictionary for configuring trailing stop loss behavior.
    -   `enabled`: Master switch for the trailing stop feature.
    -   `instrument_category_rules`: A list of rules to categorize symbols (e.g., 'forex_major', 'crypto_btc'). This allows for different trailing stop settings per asset class.
    -   `instrument_category_settings`: A dictionary mapping the categories defined in the rules to specific parameter sets (e.g., `mode`, `atr_multiplier`, `break_even_pips`). This provides highly granular and customized exit logic for different types of instruments.
