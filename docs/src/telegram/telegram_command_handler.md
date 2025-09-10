# Documentation for `src/telegram/telegram_command_handler.py`

This file defines the `TelegramCommandHandler` class, which is designed to decouple the logic of command processing from the main `TelegramBot` class. It acts as a dedicated handler for the business logic associated with each user command.

## Core Responsibilities

1.  **Command Registration**:
    -   It provides a `register_command` method to associate a command string (e.g., "status") with a specific handler function.
    -   The `register_all_commands` method is the main entry point for connecting all the defined command logic to the `TelegramBot` instance. This method injects the handlers into the `TelegramBot`'s dispatcher.

2.  **Command Logic Implementation**:
    -   It contains the implementation for complex commands that require data fetching, processing, and formatting. This includes commands like:
        -   `/history`: Fetches trade history from the `MT5Handler`, processes it, formats it into a readable summary, and handles CSV/Excel export logic.
        -   `/metrics` and `/performance`: Gathers historical deal data, calculates a wide range of performance metrics (win rate, profit factor, drawdown, etc.), and presents them in a detailed summary.
        -   `/balance`: Retrieves real-time account information (balance, equity, margin) from the `MT5Handler`.
        -   `/statustable`: Gets a list of currently open positions and formats them into a clean, tabular view.
        -   `/report`: Generates and sends a visual equity curve chart using `matplotlib`.

3.  **Callback Query Handling**:
    -   It includes the `handle_callback_query` method, which is registered with the `TelegramBot` to process interactions with inline keyboard buttons (e.g., the date range selectors for the `/history` command).

4.  **Interaction with Core Components**:
    -   It holds references to the main `TradingBot` and `MT5Handler` instances.
    -   This allows it to access the necessary data (e.g., trade history, account info) and to trigger actions (e.g., enabling/disabling trading) by calling methods on the `TradingBot`.

## Architectural Role

-   **Separation of Concerns**: By moving the command logic out of `telegram_bot.py`, this class helps to keep the `TelegramBot` class focused on its core responsibility: communication with the Telegram API. The `TelegramCommandHandler` focuses on *what to do* when a command is received.
-   **Modularity and Testability**: This separation makes the codebase cleaner and easier to maintain. The command logic can be tested more easily in isolation from the live Telegram API.
-   **Extensibility**: Adding new, complex commands is straightforward. A developer can add a new handler method to this class and register it in `register_all_commands` without needing to modify the `TelegramBot` class itself.
