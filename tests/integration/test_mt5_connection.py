import unittest
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from src.mt5_handler import MT5Handler
from config.config import MT5_CONFIG

class TestMT5Connection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mt5_handler = MT5Handler()
        
    def tearDown(self):
        """Clean up after each test method."""
        if mt5.initialize():
            mt5.shutdown()

    def test_mt5_initialization(self):
        """Test MT5 initialization."""
        self.assertTrue(self.mt5_handler.initialize())
        self.assertTrue(mt5.initialize())

    def test_mt5_login(self):
        """Test MT5 login."""
        if not mt5.initialize():
            self.skipTest("MT5 initialization failed")
            
        login_result = mt5.login(
            login=MT5_CONFIG["login"],
            password=MT5_CONFIG["password"],
            server=MT5_CONFIG["server"]
        )
        self.assertTrue(login_result)

    def test_data_retrieval(self):
        """Test market data retrieval."""
        if not mt5.initialize():
            self.skipTest("MT5 initialization failed")
            
        # Get recent EURUSD M5 data
        rates = mt5.copy_rates_from(
            "EURUSD",
            mt5.TIMEFRAME_M5,
            datetime.now() - timedelta(days=1),
            100
        )
        self.assertIsNotNone(rates)
        self.assertTrue(len(rates) > 0)

    def test_symbol_info(self):
        """Test symbol information retrieval."""
        if not mt5.initialize():
            self.skipTest("MT5 initialization failed")
            
        symbol_info = mt5.symbol_info("EURUSD")
        self.assertIsNotNone(symbol_info)
        self.assertEqual(symbol_info.name, "EURUSD")

    def test_account_info(self):
        """Test account information retrieval."""
        if not mt5.initialize():
            self.skipTest("MT5 initialization failed")
            
        account_info = mt5.account_info()
        self.assertIsNotNone(account_info)
        self.assertTrue(hasattr(account_info, 'balance'))
        self.assertTrue(hasattr(account_info, 'equity'))

    def test_market_order_validation(self):
        """Test market order parameter validation."""
        if not mt5.initialize():
            self.skipTest("MT5 initialization failed")
            
        # Prepare order request
        symbol = "EURUSD"
        lot = 0.1
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Validate order parameters
        self.assertTrue(mt5.order_check(request))

    def test_connection_stability(self):
        """Test connection stability with multiple operations."""
        if not mt5.initialize():
            self.skipTest("MT5 initialization failed")
            
        for _ in range(5):
            # Perform multiple operations
            self.assertIsNotNone(mt5.account_info())
            self.assertIsNotNone(mt5.symbol_info("EURUSD"))
            rates = mt5.copy_rates_from(
                "EURUSD",
                mt5.TIMEFRAME_M5,
                datetime.now(),
                10
            )
            self.assertIsNotNone(rates)

if __name__ == '__main__':
    unittest.main() 