import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.signal_generator import SignalGenerator
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis

@pytest.fixture
def sample_data():
    """Fixture to provide sample market data for tests."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
    return pd.DataFrame({
        'open': np.random.randn(len(dates)) + 100,
        'high': np.random.randn(len(dates)) + 101,
        'low': np.random.randn(len(dates)) + 99,
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

@pytest.fixture
def signal_generator():
    """Fixture to provide SignalGenerator instance."""
    return SignalGenerator()

@pytest.fixture
def market_analysis():
    """Fixture to provide MarketAnalysis instance."""
    return MarketAnalysis()

@pytest.fixture
def smc_analysis():
    """Fixture to provide SMCAnalysis instance."""
    return SMCAnalysis()

@pytest.fixture
def mtf_analysis():
    """Fixture to provide MTFAnalysis instance."""
    return MTFAnalysis()

@pytest.fixture
def test_config():
    """Fixture to provide test configuration."""
    return {
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'timeframes': ['M5', 'M15', 'H1', 'H4'],
        'risk_per_trade': 0.01,
        'initial_balance': 10000,
        'commission': 0.0001,
        'spread': 2
    } 