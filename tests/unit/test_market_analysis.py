import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis import MarketAnalysis

class TestMarketAnalysis(unittest.TestCase):
    def setUp(self):
        self.market_analysis = MarketAnalysis()
        # Create sample test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
        self.test_data = pd.DataFrame({
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

    def test_market_analysis(self):
        # Test the analyze method
        result = self.market_analysis.analyze(
            df=self.test_data,
            symbol="EURUSD",
            timeframe="M5"
        )
        
        # Verify the structure of the analysis result
        self.assertIsInstance(result, dict)
        self.assertIn('market_structure', result)
        self.assertIn('session_conditions', result)
        self.assertIn('swing_points', result)
        self.assertIn('order_blocks', result)
        self.assertIn('fair_value_gaps', result)
        self.assertIn('structure_breaks', result)

        # Verify market structure analysis
        self.assertIsInstance(result['market_structure'], dict)
        
        # Verify session conditions
        self.assertIsInstance(result['session_conditions'], dict)
        self.assertIn('session', result['session_conditions'])
        self.assertIn('suitable_for_trading', result['session_conditions'])
        
        # Verify swing points
        self.assertIsInstance(result['swing_points'], dict)
        self.assertIn('highs', result['swing_points'])
        self.assertIn('lows', result['swing_points'])
        
        # Verify order blocks
        self.assertIsInstance(result['order_blocks'], list)
        
        # Verify fair value gaps
        self.assertIsInstance(result['fair_value_gaps'], dict)
        self.assertIn('bullish', result['fair_value_gaps'])
        self.assertIn('bearish', result['fair_value_gaps'])
        
        # Verify structure breaks
        self.assertIsInstance(result['structure_breaks'], list)

if __name__ == '__main__':
    unittest.main() 