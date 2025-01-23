from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    signal_type = Column(String, nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    indicators = Column(String)  # JSON string of indicator values
    sentiment_score = Column(Float)
    executed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Signal(symbol='{self.symbol}', type='{self.signal_type}', timestamp='{self.timestamp}')>"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey('signals.id'))
    symbol = Column(String, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    direction = Column(String, nullable=False)  # LONG, SHORT
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    pnl = Column(Float)
    status = Column(String, default='OPEN')  # OPEN, CLOSED, CANCELLED
    notes = Column(String)
    
    signal = relationship("Signal", backref="trades")
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', direction='{self.direction}', status='{self.status}')>"

class NewsEvent(Base):
    __tablename__ = 'news_events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    impact = Column(String)  # HIGH, MEDIUM, LOW
    sentiment_score = Column(Float)
    affected_pairs = Column(String)  # JSON string of affected currency pairs
    
    def __repr__(self):
        return f"<NewsEvent(title='{self.title}', impact='{self.impact}', timestamp='{self.timestamp}')>" 