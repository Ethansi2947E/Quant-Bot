'''
Backtesting Trade Simulator

This module defines the data structure for a simulated trade, used to track
the state and outcome of trades within the backtesting engine.
'''

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

@dataclass
class SimulatedTrade:
    """A data class to hold the state of a single simulated trade."""
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    direction: str  # 'buy' or 'sell'
    symbol: str

    # --- State Tracking ---
    status: str = 'open'  # Can be 'open', 'closed'
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None  # e.g., 'TP', 'SL', 'end_of_data'

    def close(self, exit_time: datetime, exit_price: float, reason: str):
        """Closes the trade and calculates the PnL."""
        if self.status == 'closed':
            return

        self.status = 'closed'
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason

        # Calculate PnL in terms of price movement
        price_change = self.exit_price - self.entry_price

        if self.direction == 'buy':
            self.pnl = price_change * self.position_size
        else:  # sell
            self.pnl = -price_change * self.position_size

        # Note: This PnL is simplified. A more advanced implementation would
        # account for contract size, pip value, and commission.
