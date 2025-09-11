'''
Backtesting Risk Simulator

This module provides a simulated RiskManager for backtesting purposes.
It calculates position sizes based on risk parameters without any dependency
on live account information or the MT5 platform.
'''

from loguru import logger

class BacktestRiskManager:
    """Simulates risk management for backtesting."""

    def __init__(self, risk_per_trade: float = 0.01, max_lot_size: float = 10.0):
        """
        Initializes the backtest risk manager.

        Args:
            risk_per_trade (float): The fraction of the account to risk per trade (e.g., 0.01 for 1%).
            max_lot_size (float): The maximum allowed position size in lots.
        """
        self.risk_per_trade = risk_per_trade
        self.max_lot_size = max_lot_size
        logger.info(f"BacktestRiskManager initialized with {self.risk_per_trade*100}% risk per trade.")

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        contract_size: float = 100000,  # Standard lot for Forex
        margin_currency_to_usd: float = 1.0 # Assuming USD account for simplicity
    ) -> float:
        """
        Calculates the position size in lots for a trade.

        Args:
            account_balance (float): The current simulated account balance.
            entry_price (float): The entry price of the trade.
            stop_loss_price (float): The stop loss price of the trade.
            contract_size (float): The contract size of the instrument.
            margin_currency_to_usd (float): Conversion rate for the margin currency to USD.

        Returns:
            float: The calculated position size in lots.
        """
        if entry_price <= 0 or stop_loss_price <= 0 or account_balance <= 0:
            logger.warning("Cannot calculate position size with non-positive inputs.")
            return 0.0

        # 1. Calculate risk amount in account currency
        risk_amount = account_balance * self.risk_per_trade

        # 2. Calculate stop loss distance in price terms
        stop_loss_pips = abs(entry_price - stop_loss_price)
        if stop_loss_pips == 0:
            logger.warning("Stop loss distance is zero. Cannot calculate position size.")
            return 0.0

        # 3. Calculate the value of a single pip
        # This is a simplified calculation. A real implementation would need symbol-specific pip value.
        # For now, we assume a standard pip value for forex.
        value_per_pip = contract_size * 0.0001 # For most non-JPY pairs

        # 4. Calculate position size
        # (Risk Amount) / (SL pips * Value per Pip)
        position_size = risk_amount / (stop_loss_pips * value_per_pip * margin_currency_to_usd)

        # 5. Apply constraints
        position_size = round(position_size, 2)  # Round to 2 decimal places for standard lots
        position_size = min(position_size, self.max_lot_size) # Cap at max lot size

        logger.debug(
            f"Calculated position size: {position_size} lots for balance {account_balance:.2f} "
            f"with SL distance {stop_loss_pips:.5f}"
        )

        return position_size
