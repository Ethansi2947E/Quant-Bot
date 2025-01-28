import unittest
import time
import psutil
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis import MarketAnalysis
from src.signal_generator import SignalGenerator

class TestSystemPerformance(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.market_analysis = MarketAnalysis()
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

    def test_signal_generation_performance(self):
        """Test the performance of signal generation."""
        start_time = time.time()

        # Generate signals for the entire dataset
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
        self.assertIsNotNone(signal)
        self.assertIn('signal_type', signal)

    def test_market_analysis_performance(self):
        """Test the performance of market analysis."""
        start_time = time.time()
    
        # Perform market analysis
        market_context = self.market_analysis.analyze(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5'
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
        self.assertIsNotNone(market_context)
        self.assertIsInstance(market_context, dict)

    def test_complete_analysis_performance(self):
        """Test the performance of the complete analysis pipeline."""
        start_time = time.time()

        # 1. Market Analysis
        market_context = self.market_analysis.analyze(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5'
        )
        self.assertIsNotNone(market_context)

        # 2. Signal Generation
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        self.assertIsNotNone(signal)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.assertLess(execution_time, 10.0)  # Complete pipeline should finish within 10 seconds

    def test_memory_usage(self):
        """Test memory usage during analysis."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

        # Perform complete analysis
        market_context = self.market_analysis.analyze(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5'
        )
        
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        self.assertLess(memory_increase, 500)  # Memory increase should be less than 500MB

if __name__ == '__main__':
    unittest.main() 