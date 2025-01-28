from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, DeclarativeBase
from sqlalchemy.orm import relationship
from datetime import datetime

class Base(DeclarativeBase):
    pass

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
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # 'buy' or 'sell'
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)
    status = Column(String(20), nullable=False)  # 'open', 'closed', 'cancelled'
    is_successful = Column(Boolean, nullable=True)
    notes = Column(String)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    timeframe = Column(String, nullable=False)
    
    signal = relationship("Signal", backref="trades")
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', direction='{self.direction}', entry_price={self.entry_price})>"

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