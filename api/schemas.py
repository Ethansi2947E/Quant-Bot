from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime

# This file will contain the Pydantic models (schemas) that define
# the shape of the data in our API.

class TradeBase(BaseModel):
    symbol: str
    type: str
    open_price: float
    close_price: float
    profit: float

class Trade(TradeBase):
    id: int

    class Config:
        orm_mode = True

class SignalBase(BaseModel):
    symbol: str
    timeframe: str
    strategy: str
    direction: str
    price: float
    details: Optional[Dict[str, Any]] = None

class SignalCreate(SignalBase):
    pass

class Signal(SignalBase):
    id: str
    timestamp: datetime

    class Config:
        orm_mode = True

class DashboardKPIs(BaseModel):
    total_profit_loss: Any
    win_rate: Any
    total_trades: Any
    portfolio_value: Any 