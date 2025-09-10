# Documentation for `src/utils/market_utils.py`

This file contains a collection of utility functions designed to perform common, market-related calculations. These functions abstract away complex or repetitive logic, making the main application code cleaner and more readable.

## Core Functions

### Pip Calculation

Pips (Percentage in Point) are a fundamental unit of measurement in forex and CFD trading, but their price value can differ significantly between assets. These functions handle the conversion between pips and price values.

-   **`calculate_pip_value(symbol, ...)`**:
    -   **Purpose**: Determines the price value of a single pip for a given symbol. For example, for EURUSD, a pip is typically 0.0001, while for USDJPY, it's 0.01.
    -   **Logic**: It fetches the symbol's information (specifically `point` and `digits`) from the `MT5Handler` and uses the standard industry convention to calculate the pip value. It also includes fallback logic for common symbols in case live data is unavailable.

-   **`convert_pips_to_price(pips, symbol, ...)`**:
    -   A convenience function that uses `calculate_pip_value` to convert a distance measured in pips into a price difference.

-   **`convert_price_to_pips(price_diff, symbol, ...)`**:
    -   The inverse of the above, converting a price difference back into pips.

### Spread Adjustment

The spread (the difference between the bid and ask price) is a cost of trading. A signal might be generated based on the bid price, but a buy order will execute at the higher ask price. This can alter the intended risk-to-reward ratio of a trade.

-   **`adjust_trade_for_spread(...)`**:
    -   **Purpose**: To maintain the integrity of a trade's intended risk and reward by adjusting the stop-loss and take-profit levels to account for the current spread.
    -   **Logic**:
        1.  It gets the current bid and ask prices from the `MT5Handler`.
        2.  It determines the *actual* entry price based on the order type (ask for a buy, bid for a sell).
        3.  It calculates the original risk (distance from entry to stop-loss) and reward (distance from entry to take-profit) from the signal.
        4.  It then re-applies that same risk and reward distance to the *actual* entry price, yielding new, spread-adjusted stop-loss and take-profit levels.

## How It's Used

-   The pip conversion functions are used throughout the application, particularly in the `RiskManager`, to perform accurate position sizing and risk calculations.
-   `adjust_trade_for_spread` is an optional but highly recommended utility that can be called by the `SignalProcessor` just before a trade is placed to ensure the execution aligns with the strategy's original intent, despite the market spread.
