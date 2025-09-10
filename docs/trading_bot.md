# Documentation for `src/trading_bot.py`

This file contains the `TradingBot` class, the central orchestrator of the entire trading system. It integrates all other components—such as the MT5 handler, signal generators, risk manager, and notification systems—into a cohesive application.

## Core Classes

### `SignalGenerator`

-   **Purpose**: A base class that defines the interface for all trading strategies.
-   **Key Methods**:
    -   `__init__`: Initializes the generator with handlers for MT5 and risk management.
    -   `generate_signals`: The core logic method that every strategy must override. It receives market data and returns a list of potential trade signals.
-   **Inheritance**: Any new trading strategy created in the `src/strategy` directory must inherit from this class to be compatible with the bot.

### `TradingBot`

-   **Purpose**: The main class that manages the bot's lifecycle, data flow, and event loops.
-   **Architecture**: It operates as a state machine that can be started, stopped, and controlled. It initializes and holds instances of all major components.

#### Initialization (`__init__`)

1.  **Configuration**: It loads default configurations from `config/config.py` and merges them with any configuration passed during instantiation. This allows for flexible setup for different environments (e.g., live vs. backtesting).
2.  **Component Instantiation**: It creates instances of all necessary components in a specific order:
    -   `MT5Handler`: For broker communication.
    -   `RiskManager`: For risk calculations and trade validation.
    -   `DataManager`: For fetching, caching, and storing market data.
    -   `TelegramBot` and `TelegramCommandHandler`: For notifications and remote control.
    -   `PositionManager`: For managing the lifecycle of open trades (e.g., trailing stops, partial take-profits).
    -   `SignalProcessor`: For validating, executing, and logging signals.
    -   `PerformanceTracker`: For tracking and reporting on trading performance.
3.  **State Management**: It initializes various state variables, such as `self.running`, `self.shutdown_requested`, `self.trading_enabled`, and dictionaries to track ticks, candles, and signals.

#### Main Lifecycle Methods

-   **`async def start()`**:
    -   The primary method to begin the bot's operation.
    -   It initializes the MT5 connection and all other components.
    -   It performs a `_perform_startup_analysis` to "warm up" the strategies with historical data.
    -   **Crucially, it reads the `execution_mode` from the config** and starts the appropriate main event loop (`live_tick_event_loop` for 'tick' mode, `tick_event_loop` for 'bar' mode).
    -   It also starts ancillary loops for monitoring open trades (`_monitor_trades_loop`) and handling shutdown requests.
    -   Returns an `asyncio.Future` that completes when the bot has fully shut down.

-   **`async def stop()`**:
    -   Initiates a graceful shutdown of the bot.
    -   It cancels running tasks and, if configured, closes all open positions.

#### Event Loops (Execution Modes)

The bot's core logic runs in one of two event loops, determined by the `execution_mode` config:

1.  **`async def tick_event_loop()` (Bar-Close Mode)**:
    -   This is the **recommended and default** mode.
    -   It runs a high-frequency loop that repeatedly calls `_check_for_new_candle` for all symbol/timeframe pairs required by the loaded strategies.
    -   Analysis is only triggered when `_check_for_new_candle` detects that a new bar has closed.

2.  **`async def live_tick_event_loop()` (Tick Mode)**:
    -   This is the high-frequency trading mode.
    -   In addition to checking for new candles, this loop also processes every new price tick via `_process_live_tick_for_symbol`.
    -   Each new tick can potentially trigger a real-time analysis run (`run_realtime_analysis_for_symbol`), making it suitable for strategies that need to react instantly to price changes.

#### Analysis and Signal Processing

-   **`async def run_analysis_cycle_for_symbol(symbol)`**:
    -   This is the main analysis workflow, typically triggered by a new candle event.
    -   It forces a fresh data update from the broker via `DataManager`.
    -   It then calls `_execute_analysis_for_symbol`, which iterates through all loaded signal generators.

-   **`_execute_analysis_for_symbol(...)`**:
    -   Loops through each `SignalGenerator` instance and calls its `generate_signals` method, passing the relevant market data.
    -   It aggregates all signals from all strategies.

-   **`async def process_signals(signals)`**:
    -   Takes the aggregated list of signals and passes them to the `SignalProcessor` instance, which handles the final validation, risk sizing, and order execution.

#### Strategy and Symbol Loading

-   **`_load_symbols_from_config()`**: Reads the list of tradable symbols from `TRADING_CONFIG`.
-   **`_load_available_signal_generators()`**: Dynamically scans the `src/strategy` directory for any Python files.
-   **`_initialize_signal_generators()`**: It imports classes that inherit from `SignalGenerator`, instantiates them, and registers their data requirements (`required_timeframes` and `lookback_periods`) with the `DataManager`.

#### Telegram Integration

-   The class includes numerous `handle_*_command` methods that serve as callbacks for the `TelegramCommandHandler`. These methods allow a user to check status, enable/disable trading, change strategies, and shut down the bot remotely.
