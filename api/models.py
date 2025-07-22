from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, JSON
from .database import Base
from datetime import datetime, UTC
import uuid

class Trade(Base):
    __tablename__ = "trades"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket = Column(Integer, unique=True, index=True)
    symbol = Column(String, index=True)
    order_type = Column(String)
    volume = Column(Float)
    open_time = Column(DateTime, default=lambda: datetime.now(UTC))
    close_time = Column(DateTime, nullable=True)
    open_price = Column(Float)
    close_price = Column(Float, nullable=True)
    profit = Column(Float)
    status = Column(String)
    magic = Column(Integer)
    comment = Column(String)

class Signal(Base):
    __tablename__ = "signals"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    symbol = Column(String, index=True)
    timeframe = Column(String)
    strategy = Column(String) # e.g., "Supertrend"
    direction = Column(String) # "buy" or "sell"
    price = Column(Float) # Price at the time of the signal
    details = Column(JSON, nullable=True) # For extra info like indicator values 