import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the Python path to allow importing RiskManager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import the module itself to access its global _risk_manager_instance
import risk_manager as risk_manager_module 
from risk_manager import RiskManager # For type hinting and direct use of the class

class TestRiskManagerPortfolioLimits(unittest.TestCase):

    def setUp(self):
        """Set up for each test case."""
        self.mock_mt5_handler = Mock()
        
        # Reset the global singleton instance in the imported module before creating a new RiskManager
        # This ensures that each test gets a fresh RiskManager instance, 
        # bypassing the singleton logic within RiskManager.__init__ if it exists.
        risk_manager_module._risk_manager_instance = None 
        
        self.risk_manager = RiskManager(mt5_handler=self.mock_mt5_handler)


    def test_no_positions_sufficient_cash(self):
        """Test with no open positions and sufficient cash."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 10000.0}
        self.mock_mt5_handler.get_open_positions.return_value = []
        
        def mock_get_last_tick(symbol):
            if symbol == "EURUSD":
                return {'ask': 1.10000, 'bid': 1.09980, 'last': 1.10000, 'time': 1234567890}
            return None

        self.mock_mt5_handler.get_last_tick = Mock(side_effect=mock_get_last_tick)

        tickers_to_analyze = ["EURUSD"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)

        self.assertIn("EURUSD", analysis)
        eurusd_analysis = analysis["EURUSD"]
        self.assertEqual(eurusd_analysis['current_price'], 1.10000)
        # Expected: 20% of 10000 = 2000. Capped by cash (10000). So, 2000.
        self.assertAlmostEqual(eurusd_analysis['remaining_position_limit'], 2000.0, places=2)
        self.assertNotIn("error", eurusd_analysis['reasoning'])

    def test_with_existing_long_position(self):
        """Test with an existing long position affecting portfolio value and exposure."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 5000.0} # Cash
        self.mock_mt5_handler.get_open_positions.return_value = [
            {'symbol': 'EURUSD', 'volume': 1.0, 'type': 0, 'current_price': 1.10000, 'open_price': 1.08000} # type 0 = BUY
        ]
        
        def mock_get_last_tick(symbol):
            if symbol == "EURUSD": # Price for new trade analysis or re-valuation
                return {'ask': 1.10000, 'bid': 1.09980, 'last': 1.10000}
            elif symbol == "GBPUSD":
                 return {'ask': 1.25000, 'bid': 1.24980, 'last': 1.25000}
            return None
        self.mock_mt5_handler.get_last_tick = Mock(side_effect=mock_get_last_tick)

        # total_portfolio_value = cash (5000) + long_value (1.0 * 1.10000 = 110000 if notional)
        # Assuming MT5 volume 1.0 means 100,000 units.
        # Let's assume for simplicity here that current_price in get_open_positions is the valuation price.
        # So, current_market_value_of_eurusd_pos = 1.0 * 1.10000 (needs interpretation of volume)
        # For MT5, volume is in lots. If 1 lot = 100,000 units:
        # EURUSD long value = 1 * 100,000 * 1.10000 = 110,000
        # total_portfolio_value = 5000 (cash) + 110000 (EURUSD value) = 115,000
        
        # For this test, let's re-mock get_last_tick to ensure the price used for new analysis is clear.
        # The internal logic of calculate_portfolio_risk_limits uses `pos.get('current_price')` for existing positions.
        
        tickers_to_analyze = ["EURUSD", "GBPUSD"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)

        # EURUSD analysis
        self.assertIn("EURUSD", analysis)
        eurusd_analysis = analysis["EURUSD"]
        self.assertEqual(eurusd_analysis['current_price'], 1.10000) # From existing position's current_price

        # Total Portfolio Value: Cash (5000) + EURUSD value (1.0 * 1.10000 = 1.1, this seems to treat volume as direct multiplier not lots)
        # The current `calculate_portfolio_risk_limits` implementation does: market_value = volume * price_at_valuation
        # This needs to align with how MT5 lots and contract sizes work if `volume` is in lots.
        # For now, assuming the test reflects the current simple math in the method:
        # EURUSD value = 1.0 * 1.10000 = 1.1 (if volume is not in lots but direct units, which is unlikely for MT5)
        # This highlights a potential area for refinement in `calculate_portfolio_risk_limits` for realistic lot valuation.
        # Let's assume the test needs to mock a contract size or adjust expected values based on current implementation.
        
        # CURRENT IMPLEMENTATION: market_value = volume * price_at_valuation.
        # With pos_volume = 1.0, pos_price = 1.1, long_value = 1.1
        # total_portfolio_value = 5000 + 1.1 = 5001.1
        # cash_limit_per_eurusd_ticker (20%) = 5001.1 * 0.20 = 1000.22
        # current_market_value_of_eurusd_pos = 1.1
        # remaining_cash_for_new_trades_on_eurusd = 1000.22 - 1.1 = 999.12
        # final_remaining_cash_allocation (capped by cash 5000) = 999.12
        self.assertAlmostEqual(eurusd_analysis['reasoning']['total_portfolio_value'], 5000.0 + 1.10000 * 1.0, places=2)
        self.assertAlmostEqual(eurusd_analysis['remaining_position_limit'], ( (5000.0 + 1.10000 * 1.0) * 0.20) - (1.10000 * 1.0) , places=2)
        self.assertNotIn("error", eurusd_analysis['reasoning'])

        # GBPUSD analysis (no existing position)
        self.assertIn("GBPUSD", analysis)
        gbpusd_analysis = analysis["GBPUSD"]
        self.assertEqual(gbpusd_analysis['current_price'], 1.25000) # From get_last_tick
        # limit_per_gbpusd_ticker = 5001.1 * 0.20 = 1000.22
        # current_exposure_gbpusd = 0
        # remaining = 1000.22
        # final (capped by cash 5000) = 1000.22
        expected_gbpusd_limit = (5000.0 + 1.10000 * 1.0) * 0.20
        self.assertAlmostEqual(gbpusd_analysis['remaining_position_limit'], expected_gbpusd_limit, places=2)
        self.assertNotIn("error", gbpusd_analysis['reasoning'])

    def test_with_existing_short_position(self):
        """Test with an existing short position."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 10000.0} # Cash
        self.mock_mt5_handler.get_open_positions.return_value = [
            {'symbol': 'USDJPY', 'volume': 0.5, 'type': 1, 'current_price': 150.00, 'open_price': 152.00} # type 1 = SELL
        ]
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            'USDJPY': {'ask': 150.05, 'bid': 150.00, 'last': 150.00}, # New analysis price
            'EURUSD': {'ask': 1.10000, 'bid': 1.09980, 'last': 1.10000}
        }.get(symbol)

        # total_portfolio_value = cash (10000) - short_value (0.5 * 150.00 = 75) = 9925.0
        # (Assuming volume is direct units for now, per current implementation detail)
        total_portfolio_val_expected = 10000.0 - (0.5 * 150.00)
        
        tickers_to_analyze = ["USDJPY", "EURUSD"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)

        self.assertIn("USDJPY", analysis)
        usdjpy_analysis = analysis["USDJPY"]
        self.assertEqual(usdjpy_analysis['current_price'], 150.00) # From existing position current_price for valuation
        self.assertAlmostEqual(usdjpy_analysis['reasoning']['total_portfolio_value'], total_portfolio_val_expected, places=2)
        
        # limit_usdjpy = 9925.0 * 0.20 = 1985.0
        # current_exposure_usdjpy = abs(0 - (0.5 * 150.00)) = 75.0
        # remaining_usdjpy = 1985.0 - 75.0 = 1910.0
        # final_usdjpy (capped by cash 10000) = 1910.0
        expected_usdjpy_limit = (total_portfolio_val_expected * 0.20) - (0.5 * 150.00)
        self.assertAlmostEqual(usdjpy_analysis['remaining_position_limit'], expected_usdjpy_limit, places=2)

        self.assertIn("EURUSD", analysis)
        eurusd_analysis = analysis["EURUSD"]
        self.assertEqual(eurusd_analysis['current_price'], 1.10000)
        # limit_eurusd = 9925.0 * 0.20 = 1985.0
        # current_exposure_eurusd = 0
        # final_eurusd (capped by cash 10000) = 1985.0
        self.assertAlmostEqual(eurusd_analysis['remaining_position_limit'], total_portfolio_val_expected * 0.20, places=2)


    def test_insufficient_cash_for_limit(self):
        """Test when cash is less than the 20% portfolio allocation for a ticker."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 100.0} # Low cash
        self.mock_mt5_handler.get_open_positions.return_value = []
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            'BTCUSD': {'ask': 30000.00, 'bid': 29990.00, 'last': 30000.00}
        }.get(symbol)

        # total_portfolio_value = 100 (cash)
        # limit_btcusd = 100 * 0.20 = 20.0
        # current_exposure_btcusd = 0
        # remaining_btcusd = 20.0
        # final_btcusd (capped by cash 100) = 20.0
        tickers_to_analyze = ["BTCUSD"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)
        self.assertIn("BTCUSD", analysis)
        btcusd_analysis = analysis["BTCUSD"]
        self.assertEqual(btcusd_analysis['current_price'], 30000.00)
        self.assertAlmostEqual(btcusd_analysis['remaining_position_limit'], 20.0, places=2) # 0.20 * 100

    def test_ticker_price_zero_or_fail(self):
        """Test when get_last_tick fails or returns zero/invalid price for a ticker."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 10000.0}
        self.mock_mt5_handler.get_open_positions.return_value = []
        
        def mock_get_last_tick_failure(symbol):
            if symbol == "OILUSD":
                return None # Simulate failure
            if symbol == "GASUSD":
                return {'ask': 0.0, 'bid': 0.0} # Simulate zero price
            return {'ask': 1.0, 'bid': 1.0} # Default for other valid symbols

        self.mock_mt5_handler.get_last_tick = Mock(side_effect=mock_get_last_tick_failure)

        tickers_to_analyze = ["OILUSD", "GASUSD"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)

        self.assertIn("OILUSD", analysis)
        oil_analysis = analysis["OILUSD"]
        self.assertEqual(oil_analysis['current_price'], 0.0)
        self.assertEqual(oil_analysis['remaining_position_limit'], 0.0)
        self.assertIn("error", oil_analysis['reasoning'])
        self.assertEqual(oil_analysis['reasoning']['error'], "Failed to fetch valid price for limit calculation")


        self.assertIn("GASUSD", analysis)
        gas_analysis = analysis["GASUSD"]
        self.assertEqual(gas_analysis['current_price'], 0.0)
        self.assertEqual(gas_analysis['remaining_position_limit'], 0.0)
        self.assertIn("error", gas_analysis['reasoning'])
        self.assertEqual(gas_analysis['reasoning']['error'], "Failed to fetch valid price for limit calculation")

    def test_mt5_handler_unavailable(self):
        """Test when mt5_handler is None."""
        self.risk_manager.mt5_handler = None # Simulate MT5 handler not being available
        
        tickers_to_analyze = ["EURUSD"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)
        
        self.assertIn("EURUSD", analysis)
        eurusd_analysis = analysis["EURUSD"]
        self.assertEqual(eurusd_analysis['current_price'], 0.0)
        self.assertEqual(eurusd_analysis['remaining_position_limit'], 0.0)
        self.assertIn("error", eurusd_analysis['reasoning'])
        self.assertEqual(eurusd_analysis['reasoning']['error'], "MT5Handler not available")

    def test_portfolio_over_limit_for_new_trade(self):
        """Test when a ticker is already over its 20% exposure limit."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 1000.0} # Cash
        # Existing large position in AAPL
        self.mock_mt5_handler.get_open_positions.return_value = [
            {'symbol': 'AAPL', 'volume': 5.0, 'type': 0, 'current_price': 200.00} 
        ]
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            'AAPL': {'ask': 200.00, 'bid': 199.90}, # Price for valuation
        }.get(symbol)

        # Cash = 1000
        # AAPL value = 5.0 * 200.00 = 1000.0 (using current naive volume * price)
        # Total Portfolio Value = 1000 (cash) + 1000 (AAPL) = 2000.0
        # AAPL Limit (20% of TPV) = 2000.0 * 0.20 = 400.0
        # Current AAPL Exposure = 1000.0
        # Remaining for new AAPL trades = 400.0 - 1000.0 = -600.0
        # Final remaining (capped at 0 and by cash) = 0.0

        tickers_to_analyze = ["AAPL"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)

        self.assertIn("AAPL", analysis)
        aapl_analysis = analysis["AAPL"]
        self.assertEqual(aapl_analysis['current_price'], 200.00)
        self.assertAlmostEqual(aapl_analysis['remaining_position_limit'], 0.0, places=2)
        self.assertNotIn("error", aapl_analysis['reasoning']) # No error, just 0 limit
        self.assertAlmostEqual(aapl_analysis['reasoning']['final_remaining_cash_allocation_(capped_by_available_cash_and_non_negative)'], 0.0, places=2)

    def test_complex_portfolio_multiple_assets(self):
        """Test with a more complex portfolio involving multiple assets and types."""
        self.mock_mt5_handler.get_account_info.return_value = {'balance': 10000.0} # Cash
        self.mock_mt5_handler.get_open_positions.return_value = [
            {'symbol': 'EURUSD', 'volume': 1.0, 'type': 0, 'current_price': 1.10}, # Long
            {'symbol': 'USDJPY', 'volume': 0.5, 'type': 1, 'current_price': 150.0}, # Short
            {'symbol': 'GOLD', 'volume': 2.0, 'type': 0, 'current_price': 2000.0}  # Long
        ]
        self.mock_mt5_handler.get_last_tick.side_effect = lambda symbol: {
            'EURUSD': {'ask': 1.10, 'bid': 1.09},
            'USDJPY': {'ask': 150.1, 'bid': 150.0},
            'GOLD': {'ask': 2000.5, 'bid': 2000.0},
            'SILVER': {'ask': 25.0, 'bid': 24.9} # New asset to analyze
        }.get(symbol)

        # Values based on current naive (volume * price) logic:
        # Cash: 10000
        # EURUSD (long): + (1.0 * 1.10) = +1.10
        # USDJPY (short): - (0.5 * 150.0) = -75.0
        # GOLD (long): + (2.0 * 2000.0) = +4000.0
        # Total Portfolio Value = 10000 + 1.10 - 75.0 + 4000.0 = 13926.10

        total_portfolio_val_expected = 10000.0 + (1.0 * 1.10) - (0.5 * 150.0) + (2.0 * 2000.0)

        tickers_to_analyze = ["EURUSD", "USDJPY", "GOLD", "SILVER"]
        analysis = self.risk_manager.calculate_portfolio_risk_limits(tickers_to_analyze)

        # EURUSD
        eurusd_analysis = analysis["EURUSD"]
        self.assertEqual(eurusd_analysis['current_price'], 1.10)
        limit_eurusd_total = total_portfolio_val_expected * 0.20 # 2785.22
        current_exp_eurusd = 1.0 * 1.10 # 1.10
        expected_rem_eurusd = limit_eurusd_total - current_exp_eurusd # 2784.12
        self.assertAlmostEqual(eurusd_analysis['remaining_position_limit'], expected_rem_eurusd, places=2)

        # USDJPY
        usdjpy_analysis = analysis["USDJPY"]
        self.assertEqual(usdjpy_analysis['current_price'], 150.0)
        limit_usdjpy_total = total_portfolio_val_expected * 0.20 # 2785.22
        current_exp_usdjpy = 0.5 * 150.0 # 75.0
        expected_rem_usdjpy = limit_usdjpy_total - current_exp_usdjpy # 2710.22
        self.assertAlmostEqual(usdjpy_analysis['remaining_position_limit'], expected_rem_usdjpy, places=2)

        # GOLD
        gold_analysis = analysis["GOLD"]
        self.assertEqual(gold_analysis['current_price'], 2000.0)
        limit_gold_total = total_portfolio_val_expected * 0.20 # 2785.22
        current_exp_gold = 2.0 * 2000.0 # 4000.0
        expected_rem_gold = limit_gold_total - current_exp_gold # -1214.78 -> capped to 0
        self.assertAlmostEqual(gold_analysis['remaining_position_limit'], 0.0, places=2)
        
        # SILVER (new asset)
        silver_analysis = analysis["SILVER"]
        self.assertEqual(silver_analysis['current_price'], 25.0) # Ask price from get_last_tick
        limit_silver_total = total_portfolio_val_expected * 0.20 # 2785.22
        current_exp_silver = 0.0
        expected_rem_silver = limit_silver_total - current_exp_silver # 2785.22
        self.assertAlmostEqual(silver_analysis['remaining_position_limit'], expected_rem_silver, places=2)


if __name__ == '__main__':
    unittest.main() 