# Documentation for `src/utils/position_manager.py`

This file contains the `PositionManager` class, which is responsible for the active, in-flight management of open trades. While the `RiskManager` acts as a gatekeeper *before* a trade is opened, the `PositionManager` takes over *after* the trade is live.

## Core Responsibilities

1.  **Trade Lifecycle Management**:
    -   It continuously monitors all open positions fetched from the `MT5Handler`.
    -   Its primary job is to execute the trade exit strategy, which includes managing stop-losses and take-profits.

2.  **Trailing Stop Loss**:
    -   This is the most complex and critical feature of the `PositionManager`. It implements a trailing stop loss to protect profits.
    -   **How it works**: Once a trade is in profit by a certain amount (the "activation price"), the `PositionManager` will start to "trail" the stop loss behind the price. If the price moves in the trade's favor, the stop loss is moved up (for a buy) or down (for a sell). If the price reverses, the stop loss stays in place, locking in profits if the reversal continues.
    -   **Configuration**: The trailing stop logic is highly configurable via `TRADE_EXIT_CONFIG` in `config/config.py`, supporting different modes (`pips`, `ATR`, `percent`) and adaptive settings for different types of instruments.

3.  **Break-Even Stop Loss**:
    -   As a preliminary step to the trailing stop, it can move the stop loss to the entry price (plus a small buffer) once the trade has moved into profit by a predefined amount. This makes the trade "risk-free" by ensuring it cannot become a loss.

4.  **Multi-Take-Profit Management**:
    -   It handles strategies that generate multiple take-profit levels.
    -   When a price level for a partial take-profit is hit, the `PositionManager` will automatically close a portion of the trade to realize some profit, leaving the rest of the position open to run towards the next target.

5.  **Scalping Exit Logic**:
    -   For strategies identified as "scalping" strategies, it can apply a more aggressive profit-taking rule, closing the trade after it has achieved a certain percentage of its total potential profit target, rather than waiting for the full target to be hit.

## Key Features and Methods

-   **`async def manage_open_trades(self)`**:
    -   This is the main entry point for the class, called repeatedly by the `TradingBot`'s main loop.
    -   It fetches all open positions and iterates through them, applying the various management logics (multi-TP, scalping, and trailing stops).

-   **`_apply_trailing_stop(self, position)`**:
    -   Contains the detailed logic for the trailing stop.
    -   It tracks the state of each trade (e.g., has the trailing stop been activated? What is the highest/lowest price seen so far?).
    -   It calculates the new potential stop loss based on the configured mode and sends a modification request to the `MT5Handler` if the stop loss needs to be moved.

-   **`_manage_multi_tp_positions(self)`**:
    -   Handles the logic for partial take-profits. It checks the current market price against the next take-profit level for any managed positions and executes a partial close if the level is reached.

-   **`_get_instrument_config(self, symbol)`**:
    -   A sophisticated helper method that resolves the correct trailing stop configuration for a given symbol. It uses a set of rules defined in the config (e.g., `symbol_contains`, `path_starts_with`) to match a symbol to a category (like "Forex", "Volatility Index") and applies the corresponding settings.

## How It's Used

-   The `PositionManager` is instantiated by the `TradingBot`.
-   The `manage_open_trades` method is called within the bot's main `tick_event_loop`. This ensures that open positions are constantly being monitored and managed in near real-time.
-   When a trade with multiple take-profits is opened, the `SignalProcessor` calls `register_trade` to add it to the `PositionManager`'s watch list.
