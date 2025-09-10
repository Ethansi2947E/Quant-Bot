# Documentation for `src/utils/logging_setup.py`

This utility file is responsible for configuring the application's entire logging system. It uses the `Loguru` library, which provides a more powerful and developer-friendly logging experience compared to Python's standard `logging` module.

## Core Responsibilities

1.  **Centralized Configuration**: It provides a single function, `setup_logging`, that takes a configuration dictionary (`LOG_CONFIG` from `config/config.py`) to set up all logging behavior. This makes it easy to manage log levels, formats, and outputs from one place.
2.  **Console and File Logging**: It configures two primary log "sinks" (outputs):
    -   **Console (`sys.stderr`)**: Logs are always printed to the console for real-time monitoring. This can be colorized for better readability.
    -   **File**: Optionally, if `use_file_logging` is enabled in the config, logs are also written to a file. This is essential for persistent records and post-mortem analysis. The file logger supports automatic rotation, retention, and compression.
3.  **Standard Logging Interception**:
    -   A key feature of this setup is the `InterceptHandler`. This custom handler captures messages from Python's built-in `logging` module (which is used by many third-party libraries) and redirects them through `Loguru`.
    -   This ensures that **all** log messages, regardless of their origin, are formatted and handled consistently according to the `Loguru` configuration.

## Key Components

### `InterceptHandler` Class

-   A custom class that inherits from `logging.Handler`.
-   Its `emit` method is called whenever a standard library logger produces a record.
-   It takes that record, finds the appropriate `Loguru` level, and re-logs the message using the `logger.opt()` method to preserve the correct call stack information.

### `setup_logging(config: dict)` Function

-   **Purpose**: The main entry point for configuring logging. It is called once at the very beginning of the application's lifecycle in `main.py`.
-   **Workflow**:
    1.  **`logger.remove()`**: It first removes any default `Loguru` handlers to start with a clean slate.
    2.  **Configure Console Sink**: It adds the console logger, setting its level, format, and colorization based on the `config` dictionary.
    3.  **Configure File Sink (Optional)**: If enabled, it creates the log directory and adds the file logger with its specific settings (path, rotation, etc.).
    4.  **`logging.basicConfig(...)`**: This is the crucial step that installs the `InterceptHandler`, effectively hijacking the standard logging system.
    5.  It logs a few initial messages to confirm that the configuration was successful and to indicate the active log level.

## How It's Used

-   The `LOG_CONFIG` dictionary is defined in `config/config.py`.
-   The `setup_logging(LOG_CONFIG)` function is called in `main.py` immediately after loading the configuration and before any other part of the application is initialized.
-   Throughout the rest of the codebase, any file can simply do `from loguru import logger` and use the `logger` object directly. All messages will be automatically handled by the configuration set up here.
