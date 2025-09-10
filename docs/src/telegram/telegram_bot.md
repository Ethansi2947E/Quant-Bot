# Documentation for `src/telegram/telegram_bot.py`

This file contains the `TelegramBot` class, which manages all interactions with the Telegram Bot API. It is designed as a singleton to ensure there is only one instance of the bot running throughout the application.

## Core Responsibilities

1.  **Initialization and Connection**:
    -   Handles the connection to the Telegram API using the bot token from the configuration.
    -   Manages the application lifecycle, including starting, stopping, and handling connection retries with exponential backoff.
    -   Verifies the bot's identity (`get_me`) and checks that it can communicate with the `allowed_users` specified in the config.

2.  **Command Handling**:
    -   Sets up handlers for built-in commands like `/start`, `/status`, `/enable`, `/disable`, `/help`, and `/metrics`.
    -   Provides a mechanism (`register_command_handler`) for other components (like `TelegramCommandHandler`) to register custom commands dynamically.

3.  **Message and Error Handling**:
    -   Includes a generic message handler (`_message_handler`) to process user input, including keyboard button presses.
    -   Implements an error handler (`_error_handler`) to catch and log exceptions from the `python-telegram-bot` library, and to notify users of errors when possible.

4.  **User Authorization**:
    -   Strictly enforces that only users whose IDs are in the `allowed_user_ids` list can interact with the bot. Unauthorized attempts are logged and rejected.

5.  **Notifications and Alerts**:
    -   Provides a suite of methods to send formatted messages to users, including:
        -   `send_trade_alert`: For new trade signals.
        -   `send_trade_update`: For trade lifecycle events (opened, closed, modified).
        -   `send_trade_error_alert`: For failures in trade execution.
        -   `send_performance_update`: For periodic performance summaries.
        -   `notify_error`: For general bot errors.
        -   `send_management_alert`: For system-level alerts (e.g., warnings, successes).

6.  **Interactive UI (Keyboards)**:
    -   Generates and displays a dynamic, multi-level command keyboard (`ReplyKeyboardMarkup`) to make the bot easy to use without memorizing commands.
    -   Handles callback queries from inline keyboard buttons (`InlineKeyboardMarkup`), such as those used for selecting a date range for trade history.

7.  **State Management**:
    -   Tracks the bot's running state (`is_running`).
    -   Manages the trading enabled/disabled state (`trading_enabled`), which can be controlled via commands.
    -   Maintains a simple in-memory cache of recent trade history and performance metrics for quick access by commands like `/metrics`.

## Singleton Pattern

-   The file uses a global variable `_telegram_bot_instance` and a class method `get_instance()` to ensure that only one `TelegramBot` object is ever created. This is crucial for preventing multiple conflicting connections to the Telegram API. Any part of the application that needs to interact with the bot can get the single, shared instance by calling `TelegramBot.get_instance()`.
