import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.signal_generator import SignalGenerator

class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.signal_generator = SignalGenerator()
        
        # Create sample test data with proper OHLC relationships
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
        n = len(dates)
        
        # Generate base prices
        base_prices = 100 + np.random.randn(n)
        
        # Generate OHLC maintaining proper relationships
        opens = base_prices + np.random.randn(n) * 0.1
        highs = opens + np.abs(np.random.randn(n) * 0.2)
        lows = opens - np.abs(np.random.randn(n) * 0.2)
        closes = opens + np.random.randn(n) * 0.1
        
        # Adjust to maintain OHLC relationships
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        self.test_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)

    def test_signal_generation(self):
        """Test basic signal generation functionality."""
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        self.assertIsInstance(signal, dict)
        self.assertIn('signal_type', signal)
        self.assertIn('confidence', signal)
        self.assertIn('current_price', signal)
        self.assertIn('support', signal)
        self.assertIn('resistance', signal)
        self.assertIn('trend', signal)

    def test_signal_validation(self):
        """Test signal validation logic."""
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        self.assertIsInstance(signal, dict)
        self.assertIn('signal_type', signal)
        self.assertTrue(signal['signal_type'] in ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(signal['confidence'], 0)
        self.assertLessEqual(signal['confidence'], 100)

    def test_signal_strength(self):
        """Test signal strength calculation."""
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        self.assertIsInstance(signal, dict)
        self.assertIn('signal_type', signal)
        self.assertIn('confidence', signal)
        self.assertGreaterEqual(signal['confidence'], 0)
        self.assertLessEqual(signal['confidence'], 100)

if __name__ == '__main__':
    unittest.main() 