import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis import MarketAnalysis
from src.signal_generator import SignalGenerator
from src.models import Trade

class TestTradingFlow(unittest.TestCase):
    def setUp(self):
        self.market_analysis = MarketAnalysis()
        self.signal_generator = SignalGenerator()
        
        # Create sample test data
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

    def test_complete_analysis_flow(self):
        """Test the complete analysis flow from market analysis to signal generation."""
        # 1. Market Analysis
        market_context = self.market_analysis.analyze(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5'
        )
        self.assertIsNotNone(market_context)
        self.assertIsInstance(market_context, dict)
        
        # 2. Signal Generation
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        self.assertIsNotNone(signal)
        self.assertIsInstance(signal, dict)
        self.assertIn('signal_type', signal)
        self.assertIn('confidence', signal)

    def test_signal_to_trade_flow(self):
        """Test the flow from signal generation to trade creation."""
        # Generate signal
        signal = self.signal_generator.generate_signal(
            df=self.test_data,
            symbol='EURUSD',
            timeframe='M5',
            mtf_data={}
        )
        self.assertIsNotNone(signal)
        
        # Create trade if signal is actionable
        if signal['signal_type'] in ['BUY', 'SELL']:
            trade = Trade(
                symbol='EURUSD',
                direction=signal['signal_type'],
                entry_price=signal['current_price'],
                stop_loss=signal.get('support', signal['current_price'] * 0.99),
                take_profit=signal.get('resistance', signal['current_price'] * 1.01),
                timestamp=datetime.utcnow(),
                timeframe='M5'
            )
            self.assertIsInstance(trade, Trade)
            self.assertEqual(trade.symbol, 'EURUSD')
            self.assertEqual(trade.direction, signal['signal_type'])

if __name__ == '__main__':
    unittest.main() 