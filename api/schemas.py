from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, Dict, List
from datetime import datetime

# This file will contain the Pydantic models (schemas) that define
# the shape of the data in our API.

class Trade(BaseModel):
    id: str
    symbol: str
    order_type: str
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    volume: float
    profit: float
    
    model_config = ConfigDict(from_attributes=True)


class TradeHistoryResponse(BaseModel):
    trades: List[Trade]
    total_trades: int


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