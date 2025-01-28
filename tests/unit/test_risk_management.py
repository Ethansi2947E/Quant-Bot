import unittest
from datetime import datetime, UTC
from src.risk_manager import RiskManager
from src.models import Trade
from config.config import TRADING_CONFIG

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager()
        self.account_balance = 10000.0
        self.risk_per_trade = TRADING_CONFIG.get('risk_per_trade', 0.02)  # 2% risk per trade
        self.max_position_size = TRADING_CONFIG.get('max_position_size', 1.0)  # 1.0 lot
        
        # Create a sample trade
        self.test_trade = Trade(
            symbol='EURUSD',
            direction='BUY',
            entry_price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2100,
            timestamp=datetime.now(UTC),
            timeframe='M5'
        )

    def test_position_size_calculation(self):
        """Test if position size calculation respects risk limits."""
        position_size = self.risk_manager.calculate_position_size(
            account_balance=self.account_balance,
            risk_amount=self.account_balance * self.risk_per_trade,
            entry_price=self.test_trade.entry_price,
            stop_loss=self.test_trade.stop_loss,
            symbol=self.test_trade.symbol
        )
        
        self.assertIsNotNone(position_size)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.max_position_size)

    def test_risk_per_trade_limit(self):
        """Test if risk per trade is within limits."""
        risk_amount = self.risk_manager.calculate_risk_amount(
            account_balance=self.account_balance,
            risk_percentage=self.risk_per_trade
        )
        
        max_risk_amount = self.account_balance * TRADING_CONFIG.get('max_risk_per_trade', 0.03)
        self.assertLessEqual(risk_amount, max_risk_amount)

    def test_max_daily_risk(self):
        """Test if daily risk limit is respected."""
        daily_risk = self.risk_manager.calculate_daily_risk(
            account_balance=self.account_balance,
            open_trades=[self.test_trade],
            pending_trades=[self.test_trade]  # Simulate another trade
        )
        
        max_daily_risk = self.account_balance * TRADING_CONFIG.get('max_daily_risk', 0.06)
        self.assertLessEqual(daily_risk, max_daily_risk)

    def test_max_concurrent_trades(self):
        """Test if maximum concurrent trades limit is respected."""
        max_trades = TRADING_CONFIG.get('max_concurrent_trades', 3)
        can_open_trade = self.risk_manager.can_open_new_trade(
            current_trades=[self.test_trade] * (max_trades - 1)
        )
        self.assertTrue(can_open_trade)
        
        # Should not allow new trade when at max
        can_open_trade = self.risk_manager.can_open_new_trade(
            current_trades=[self.test_trade] * max_trades
        )
        self.assertFalse(can_open_trade)

    def test_trailing_stop(self):
        """Test trailing stop calculation."""
        # Simulate price movement
        current_price = 1.2050  # Price moved 50 pips in favor
        new_stop_loss = self.risk_manager.calculate_trailing_stop(
            trade=self.test_trade,
            current_price=current_price
        )
        
        # Should move stop loss up while maintaining the original distance
        original_distance = self.test_trade.entry_price - self.test_trade.stop_loss
        new_distance = current_price - new_stop_loss
        self.assertAlmostEqual(original_distance, new_distance, places=4)
        self.assertGreater(new_stop_loss, self.test_trade.stop_loss)

if __name__ == '__main__':
    unittest.main() 