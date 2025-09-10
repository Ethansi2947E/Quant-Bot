# Documentation for `src/utils/performance_tracker.py`

This file defines the `PerformanceTracker` class, a utility dedicated to calculating, tracking, and reporting on the trading bot's performance. It provides the data needed to understand the bot's profitability, risk profile, and overall effectiveness.

## Core Responsibilities

1.  **Metric Calculation**:
    -   It fetches the historical trade data from the `MT5Handler`.
    -   It processes this history to calculate a wide range of standard performance metrics, including:
        -   Total Profit/Loss
        -   Win Rate
        -   Profit Factor (Gross Profit / Gross Loss)
        -   Average Profit per Winning Trade
        -   Average Loss per Losing Trade
        -   Maximum and Current Drawdown

2.  **Per-Strategy Tracking**:
    -   A key feature is its ability to track performance metrics on a **per-strategy basis**.
    -   It uses the `comment` field of a trade (which is set to the strategy's name during signal generation) to group trades.
    -   This allows for a granular analysis of which strategies are performing well and which are underperforming. It also calculates a `"Global"` aggregate of all strategies combined.

3.  **Time-Based Performance**:
    -   It aggregates performance data over different time horizons, calculating the total profit/loss and number of trades for each day, week, and month.

4.  **Report Generation**:
    -   The `generate_performance_report` method consolidates all the calculated metrics into a single, comprehensive report dictionary. This report can then be used by other parts of the system, such as the Telegram bot or a web API, to display performance data.

## Key Features and Methods

-   **`__init__(self, mt5_handler)`**: Initializes the tracker, taking an `MT5Handler` instance to access trade history. It sets up the nested dictionary structure (`self.metrics`) to hold data for "Global" and each individual strategy.

-   **`async def update_performance_metrics(self)`**:
    -   This is the main data processing method.
    -   It calls `self.mt5_handler.get_order_history()` to get the raw trade data.
    -   It iterates through the trades, grouping them by the strategy name found in the trade's comment.
    -   It calculates the core metrics for each strategy and updates the `self.metrics` dictionary.
    -   It also performs a separate calculation for drawdown by fetching the account's equity history.

-   **`_get_new_metrics_dict(self)`**: A helper method that returns a clean, zeroed-out dictionary template for storing a strategy's metrics. This is used to initialize the tracking for "Global" and any new strategies that are discovered in the trade history.

-   **`async def generate_performance_report(self)`**:
    -   First, it calls `update_performance_metrics` to ensure all data is up-to-date.
    -   Then, it assembles the final report dictionary, structuring the overall metrics, daily/weekly/monthly performance, and other relevant data points.

## How It's Used

-   The `PerformanceTracker` is instantiated by the `TradingBot`.
-   It is primarily used by components that need to report on the bot's status. For example, the `TelegramCommandHandler` might have a `/performance` command that calls `generate_performance_report` and formats the result into a message for the user.
-   It can also be used by the `TradingBot` itself to make high-level decisions, such as disabling a strategy that is performing poorly over a sustained period.
