import unittest
from unittest.mock import Mock
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from risk_manager import RiskManager
import risk_manager as risk_manager_module # To reset singleton

class TestRiskManagerIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_mt5_handler = Mock()
        risk_manager_module._risk_manager_instance = None # Reset singleton
        self.risk_manager = RiskManager(mt5_handler=self.mock_mt5_handler)

        # Default mock behaviors that can be overridden in specific tests
        self.mock_mt5_handler.get_symbol_min_lot_size.return_value = 0.01
        # Simple normalization: round to 2 decimal places, ensure non-negative
        self.mock_mt5_handler.normalize_volume.side_effect = lambda sym, vol: round(max(0, vol), 2) 

    def test_trade_downsized_by_portfolio_limit(self):
        """Test a trade that gets downsized due to portfolio cash limit for the symbol."""
        # --- Store original RiskManager settings to restore later --- 
        original_use_fixed = self.risk_manager.use_fixed_lot_size
        original_fixed_lot = self.risk_manager.fixed_lot_size
        original_max_lot = self.risk_manager.max_lot_size

        # --- Configure RiskManager for this test case --- 
        self.risk_manager.use_fixed_lot_size = True
        self.risk_manager.fixed_lot_size = 5.0 # Initial large lot size
        self.risk_manager.max_lot_size = 10.0    # Ensure max_lot_size doesn't cap before portfolio limit

        # --- Mock MT5 Handler --- 
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 10.0}
        self.mock_mt5_handler.get_open_positions.return_value = [
            {'symbol': 'USDJPY', 'volume': 1.0, 'type': 0, 'current_price': 5.0}
        ]
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            "EURUSD": {'ask': 1.10000, 'bid': 1.09980},
            "USDJPY": {'ask': 5.0, 'bid': 4.90}
        }.get(symbol)
        self.mock_mt5_handler.get_symbol_min_lot_size.return_value = 0.01 # For EURUSD

        # --- Trade Signal --- 
        trade_signal = {
            'symbol': 'EURUSD', 'direction': 'BUY',
            'entry_price': 1.10000, 'stop_loss': 1.09000, 'take_profit': 1.12000,
        }
        
        # --- Portfolio Calculation Expectations for this setup ---
        # Cash = 10.0
        # USDJPY value = 1.0 * 5.0 = 5.0
        # Total Portfolio Value (TPV) = 10.0 + 5.0 = 15.0
        # EURUSD Cash Limit (20% of TPV) = 15.0 * 0.20 = 3.0
        # Current EURUSD Exposure = 0
        # Remaining cash for new EURUSD = 3.0
        # Max_pseudo_lots for EURUSD (cash_limit / price) = 3.0 / 1.10000 = 2.7272...
        # Fixed lot size set to 5.0. max_lot_size set to 10.0. So initial position_size will be 5.0.
        # Since 5.0 (initial lots) > 2.7272 (portfolio-derived max pseudo-lots), it will be capped.
        # Capped position_size = 2.7272...
        # Normalized: round(max(0, 2.7272...), 2) = 2.73.
        # Min lot for EURUSD is 0.01. Since 2.73 >= 0.01, it's valid.
        # Expected final size: 2.73
        
        validation_result = self.risk_manager.validate_trade(trade_signal, 10.0, self.mock_mt5_handler.get_open_positions()) 
        
        self.assertTrue(validation_result['valid'], f"Validation failed: {validation_result.get('reason')}")
        self.assertAlmostEqual(validation_result['adjusted_position_size'], 2.73, places=2)
        # Ensure the reason mentions portfolio limit if it was the cause of adjustment
        # A more robust check might involve inspecting logs or a more detailed reason string if implemented.
        # For now, we check if the size was indeed adjusted as expected. 

        # --- Restore original RiskManager settings --- 
        self.risk_manager.use_fixed_lot_size = original_use_fixed
        self.risk_manager.fixed_lot_size = original_fixed_lot
        self.risk_manager.max_lot_size = original_max_lot

    def test_trade_invalidated_by_portfolio_limit_too_small(self):
        """Test a trade invalidated as portfolio limit makes it smaller than min_lot_size."""
        original_use_fixed = self.risk_manager.use_fixed_lot_size
        original_fixed_lot = self.risk_manager.fixed_lot_size
        original_max_lot = self.risk_manager.max_lot_size

        self.risk_manager.use_fixed_lot_size = True
        self.risk_manager.fixed_lot_size = 1.0 # Requested size
        self.risk_manager.max_lot_size = 10.0    # Ensure not capped by this

        self.mock_mt5_handler.get_account_info.return_value = {'balance': 0.25} # Very low cash
        self.mock_mt5_handler.get_open_positions.return_value = [] 
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            "EURUSD": {'ask': 1.10000, 'bid': 1.09980}
        }.get(symbol)
        self.mock_mt5_handler.get_symbol_min_lot_size.return_value = 0.10 # Higher min lot for EURUSD

        trade_signal = {
            'symbol': 'EURUSD', 'direction': 'BUY',
            'entry_price': 1.10000, 'stop_loss': 1.09000, 'take_profit': 1.12000,
        }

        # TPV = 0.25 (cash only)
        # EURUSD Limit (cash) = 0.25 * 0.20 = 0.05
        # Max_pseudo_lots for EURUSD = 0.05 / 1.10000 = 0.04545...
        # Initial fixed_lot_size = 1.0. max_lot_size = 10.0. So initial position_size = 1.0
        # Since 1.0 (initial lots) > 0.04545 (portfolio-derived max pseudo-lots), it will be capped.
        # Capped position_size = 0.04545...
        # Normalized: round(max(0, 0.04545...), 2) = 0.05.
        # Min lot for EURUSD is 0.10. Since 0.05 < 0.10, trade should be invalid.
        
        validation_result = self.risk_manager.validate_trade(trade_signal, 0.25, [])

        self.assertFalse(validation_result['valid'])
        self.assertIn("below symbol min lot", validation_result.get('reason', "").lower())
        
        self.risk_manager.use_fixed_lot_size = original_use_fixed
        self.risk_manager.fixed_lot_size = original_fixed_lot
        self.risk_manager.max_lot_size = original_max_lot

    def test_trade_unaffected_by_portfolio_limit(self):
        """Test a trade that is well within all limits."""
        original_use_fixed = self.risk_manager.use_fixed_lot_size
        original_fixed_lot = self.risk_manager.fixed_lot_size
        original_max_lot = self.risk_manager.max_lot_size

        self.risk_manager.use_fixed_lot_size = True
        self.risk_manager.fixed_lot_size = 0.5 # Small, valid lot size
        self.risk_manager.max_lot_size = 1.0 # Default max lot

        self.mock_mt5_handler.get_account_info.return_value = {'balance': 100000.0} # Ample cash
        self.mock_mt5_handler.get_open_positions.return_value = []
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            "EURUSD": {'ask': 1.10000, 'bid': 1.09980}
        }.get(symbol)
        self.mock_mt5_handler.get_symbol_min_lot_size.return_value = 0.01 # For EURUSD

        trade_signal = {
            'symbol': 'EURUSD', 'direction': 'BUY',
            'entry_price': 1.10000, 'stop_loss': 1.09000, 'take_profit': 1.12000,
        }

        # TPV = 100000
        # EURUSD Limit (cash) = 100000 * 0.20 = 20000
        # Max_pseudo_lots = 20000 / 1.1 = 18181.81...
        # Initial fixed_lot_size = 0.5. max_lot_size = 1.0. So initial position_size = 0.5.
        # Since 0.5 (initial lots) < 18181.81 (portfolio-derived max pseudo-lots), it is NOT capped by portfolio.
        # Expected final size: 0.5
        
        validation_result = self.risk_manager.validate_trade(trade_signal, 100000.0, [])
        self.assertTrue(validation_result['valid'])
        self.assertAlmostEqual(validation_result['adjusted_position_size'], 0.50, places=2)
        # Check that the reason does not mention portfolio limit if not applied explicitly
        # The reason string might be generic like "Trade meets all risk management criteria"
        # or specifically state it passed portfolio checks. For now, ensure it's valid and size is correct.
        # A more specific check would be to inspect the logs or the reasoning dict if it was returned by validate_trade.
        
        self.risk_manager.use_fixed_lot_size = original_use_fixed
        self.risk_manager.fixed_lot_size = original_fixed_lot
        self.risk_manager.max_lot_size = original_max_lot

if __name__ == '__main__':
    unittest.main() 