# Documentation for `strategy_template.py`

This file provides a foundational template for creating new trading strategies. It is designed to ensure that any new strategy integrates seamlessly with the bot's core systems, including data handling, signal processing, and risk management.

## Purpose and Usage

To create a new trading strategy, developers should:

1.  **Copy and Rename**: Make a copy of this file in the `src/strategy/` directory and give it a descriptive name (e.g., `my_new_strategy.py`).
2.  **Rename the Class**: Change the class name from `StrategyTemplate` to a unique name for the new strategy (e.g., `MyNewStrategy`).
3.  **Inherit from `SignalGenerator`**: The class must inherit from the `SignalGenerator` base class, which is defined in `src/trading_bot.py`.
4.  **Implement Core Methods**: Fill in the logic for the essential methods, especially `generate_signals`.

## Key Components of a Strategy Class

### `__init__(self, ...)`

-   **Purpose**: The constructor for the strategy.
-   **Actions**:
    1.  Call the parent class constructor: `super().__init__(**kwargs)`.
    2.  Define strategy identity: `self.name`, `self.description`.
    3.  Set essential parameters, especially the timeframes it will operate on (`primary_timeframe`, `higher_timeframe`, etc.).
    4.  Define any custom parameters the strategy requires (e.g., indicator periods, thresholds).
    5.  (Optional but Recommended) Load a timeframe-specific profile using `_load_timeframe_profile` to make the strategy adaptable.

### `@property required_timeframes(self) -> List[str]`

-   **Purpose**: This is a **critical property** that tells the `TradingBot` which timeframes of market data the strategy needs to function.
-   **Implementation**: It should return a list of strings, where each string is a valid timeframe (e.g., `"M5"`, `"H1"`). The `TradingBot` uses this list to fetch and provide the correct data `DataFrame`s to the `generate_signals` method.

### `async def generate_signals(self, market_data, **kwargs) -> List[Dict]`

-   **Purpose**: This is the **most important method** in any strategy. It contains the core logic for analyzing market data and deciding whether to issue a trade signal.
-   **Workflow**:
    1.  **Receive Data**: It receives a `market_data` dictionary, which is structured as `{ "SYMBOL": { "TIMEFRAME": pd.DataFrame } }`.
    2.  **Validate Data**: The first step should always be to check that the required `DataFrame`s exist and are not empty.
    3.  **Analyze**: Implement the strategy's logic. The template suggests a structured approach:
        -   Use a higher timeframe for trend context (`_determine_trend`).
        -   Identify key price levels (`_find_key_levels`).
        -   Look for a specific entry setup on the primary timeframe (`_check_entry_condition`).
        -   Wait for a confirmation signal (`_find_confirmation_signal`).
    4.  **Assemble Signal**: If all conditions are met, create a `signal` dictionary. This dictionary has a specific required structure, including `"symbol"`, `"direction"`, `"entry_price"`, `"stop_loss"`, and `"take_profit"`.
    5.  **Validate with RiskManager**: Before finalizing, the signal must be passed to the `RiskManager` via `rm.validate_and_size_trade(signal_details)`. This is a crucial step that checks the trade against global risk rules and calculates the final, safe position size.
    6.  **Return Valid Signals**: If the `RiskManager` approves the trade, append its returned `final_trade_params` to the list of signals to be returned.

### Helper Methods (`_determine_trend`, `_find_key_levels`, etc.)

-   **Purpose**: The template includes several placeholder "private" helper methods. It is a best practice to break down the complex logic of a strategy into smaller, single-purpose, and easily testable functions like these.

## Timeframe Profiles Pattern

-   The template introduces a powerful pattern using `TIMEFRAME_PROFILES`. This dictionary allows a developer to define different sets of parameters (like indicator periods or lookback windows) for different timeframes.
-   The `_load_timeframe_profile` method automatically selects the correct set of parameters based on the strategy's `primary_timeframe`, making the strategy highly adaptable without needing to change the code.
