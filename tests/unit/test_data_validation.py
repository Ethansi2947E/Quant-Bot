import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample test data with proper OHLC relationships
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
        n = len(dates)
        
        # Generate base prices
        base_prices = 100 + np.random.randn(n)
        
        # Generate OHLC maintaining proper relationships
        opens = base_prices + np.random.randn(n) * 0.1
        highs = opens + np.abs(np.random.randn(n) * 0.2)  # Always higher than open
        lows = opens - np.abs(np.random.randn(n) * 0.2)   # Always lower than open
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

    def test_data_structure(self):
        """Test if the data has the correct structure."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, self.test_data.columns)

    def test_data_types(self):
        """Test if the data has correct types."""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            self.assertTrue(np.issubdtype(self.test_data[col].dtype, np.number))

    def test_price_relationships(self):
        """Test if price relationships are valid."""
        # High should be highest price
        self.assertTrue(all(self.test_data['high'] >= self.test_data['open']))
        self.assertTrue(all(self.test_data['high'] >= self.test_data['close']))
        
        # Low should be lowest price
        self.assertTrue(all(self.test_data['low'] <= self.test_data['open']))
        self.assertTrue(all(self.test_data['low'] <= self.test_data['close']))
        
        # High should be greater than or equal to low
        self.assertTrue(all(self.test_data['high'] >= self.test_data['low']))

    def test_volume_validity(self):
        """Test if volume data is valid."""
        self.assertTrue(all(self.test_data['volume'] >= 0))

    def test_timestamp_continuity(self):
        """Test if timestamps are continuous."""
        time_diff = self.test_data.index.to_series().diff()[1:]
        expected_diff = pd.Timedelta(minutes=5)
        self.assertTrue(all(time_diff == expected_diff))

if __name__ == '__main__':
    unittest.main() 