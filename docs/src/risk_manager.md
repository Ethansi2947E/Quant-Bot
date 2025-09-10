# Documentation for `src/risk_manager.py`

This file contains the `RiskManager` class, a critical component responsible for enforcing all risk management rules. It acts as a gatekeeper for trades, ensuring that no single trade or the overall portfolio exposure exceeds predefined limits.

## Core Responsibilities

1.  **Position Sizing**: Calculates the appropriate trade volume (lot size) based on the account balance, risk per trade, and the stop-loss distance of a potential trade.
2.  **Trade Validation**: Before a trade is executed, it is validated against a comprehensive set of rules to ensure it aligns with the bot's risk appetite.
3.  **Portfolio-Level Risk**: Assesses the overall risk of the portfolio, including checks for maximum concurrent trades and (in future versions) concentration risk via correlation.
4.  **Dynamic Risk Adjustment**: Can dynamically adjust the risk per trade based on recent performance (e.g., increasing risk during winning streaks and decreasing it during drawdowns).
5.  **State Tracking**: Maintains statistics on daily performance, including profit/loss, trade count, and drawdown.

## Key Features and Methods

### Singleton Pattern

-   **`get_instance()`**: Like the `MT5Handler`, the `RiskManager` is a singleton to ensure a single, consistent source of risk policy throughout the application.

### Initialization (`__init__`)

-   Loads all risk parameters from the `RISK_MANAGER_CONFIG` and `TRADING_CONFIG` dictionaries in `config/config.py`. This includes:
    -   `max_risk_per_trade`
    -   `max_concurrent_trades`
    -   `min_risk_reward`
    -   `max_daily_loss`
    -   `use_fixed_lot_size` / `fixed_lot_size`

### Core Methods

-   **`calculate_position_size(...)`**:
    -   This is the heart of the risk management logic.
    -   If `use_fixed_lot_size` is true, it returns the configured fixed lot size (while respecting broker minimums).
    -   Otherwise, it calculates the lot size required to risk a specific percentage of the account balance (`max_risk_per_trade`) given the entry and stop-loss prices.
    -   It uses the `MT5Handler` to fetch symbol-specific information like point value and contract size for accurate calculations.

-   **`validate_trade(...)`**:
    -   A comprehensive pre-trade checklist. It is called by the `SignalProcessor` before any trade is placed.
    -   It performs several crucial checks:
        1.  **Parameter Validation**: Ensures the trade signal contains all necessary information (symbol, prices, direction).
        2.  **R:R Ratio**: Verifies that the trade's potential reward-to-risk ratio meets the `min_risk_reward` requirement.
        3.  **SL/TP Placement**: Confirms that the stop-loss and take-profit are on the correct side of the entry price for the given trade direction.
        4.  **Max Concurrent Trades**: Rejects the trade if the number of open positions is already at the `max_concurrent_trades` limit.
        5.  **Portfolio Limits**: Calls `calculate_portfolio_risk_limits` to check if the new trade would exceed the maximum allowed exposure for that specific asset.
    -   Returns a dictionary indicating if the trade is valid, the reason for the decision, and the calculated (or validated) position size.

-   **`calculate_portfolio_risk_limits(...)`**:
    -   Performs a portfolio-level analysis to determine the maximum position value allowed for a given ticker.
    -   It calculates the total portfolio value (cash + market value of open positions) and allocates a maximum exposure percentage per ticker (e.g., 20%).
    -   This prevents over-concentration in a single asset.

### Custom Exceptions

The file defines several custom exceptions (`InsufficientBalanceError`, `InvalidRiskParameterError`, `RiskCalculationError`) to provide clear and specific error messages when risk-related calculations or validations fail.

## How It's Used

The `RiskManager` is instantiated by the `TradingBot` and is used primarily by the `SignalProcessor`. When a new signal is generated, the `SignalProcessor` passes it to the `RiskManager`'s `validate_and_size_trade` method. Only if the trade is approved by the `RiskManager` will the `SignalProcessor` proceed to execute it via the `MT5Handler`.
