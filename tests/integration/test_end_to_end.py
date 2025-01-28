import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.signal_generator import SignalGenerator
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.mt5_handler import MT5Handler
from src.telegram_bot import TelegramBot
from src.models import Trade, Signal
from config.config import TRADING_CONFIG, MT5_CONFIG

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize components
        self.mt5_handler = MT5Handler()
        self.signal_generator = SignalGenerator()
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.telegram_bot = TelegramBot()
        
        # Test data setup
        self.symbol = "EURUSD"
        self.timeframe = "M5"
        self.test_chat_id = "123456789"

    def test_complete_trading_cycle(self):
        """Test a complete trading cycle from data retrieval to trade execution."""
        try:
            # 1. MT5 Connection and Data Retrieval
            self.assertTrue(self.mt5_handler.initialize())
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            historical_data = self.mt5_handler.get_historical_data(
                self.symbol,
                self.timeframe,
                start_date,
                end_date
            )
            self.assertIsNotNone(historical_data)
            self.assertTrue(len(historical_data) > 0)
            
            # 2. Market Analysis
            market_context = self.market_analysis.analyze(
                df=historical_data,
                symbol=self.symbol,
                timeframe=self.timeframe
            )
            self.assertIsNotNone(market_context)
            
            # 3. Signal Generation
            signal = self.signal_generator.generate_signal(
                df=historical_data,
                symbol=self.symbol,
                timeframe=self.timeframe,
                mtf_data={}
            )
            self.assertIsNotNone(signal)
            self.assertIn('signal_type', signal)
            
            # 4. Trade Creation
            if signal['signal_type'] in ['BUY', 'SELL']:
                trade = Trade(
                    symbol=self.symbol,
                    direction=signal['signal_type'],
                    entry_price=signal['current_price'],
                    stop_loss=signal.get('support', signal['current_price'] * 0.99),
                    take_profit=signal.get('resistance', signal['current_price'] * 1.01),
                    timestamp=datetime.utcnow(),
                    timeframe=self.timeframe
                )
                
                # 5. Risk Validation
                account_info = self.mt5_handler.get_account_info()
                if account_info:
                    risk_amount = account_info['balance'] * TRADING_CONFIG["risk_per_trade"]
                    position_size = 0.01  # Minimum lot size for testing
                    self.assertGreater(position_size, 0)
                    
                    # 6. Order Placement
                    order_result = self.mt5_handler.place_market_order(
                        symbol=trade.symbol,
                        order_type=trade.direction,
                        volume=position_size,
                        stop_loss=trade.stop_loss,
                        take_profit=trade.take_profit,
                        comment="Test trade"
                    )
                    if order_result:
                        self.assertIsInstance(order_result, dict)
                        self.assertIn('ticket', order_result)
                        
                        # 7. Telegram Notification
                        self.telegram_bot.send_trade_alert(
                            chat_id=self.test_chat_id,
                            symbol=trade.symbol,
                            direction=trade.direction,
                            entry=trade.entry_price,
                            sl=trade.stop_loss,
                            tp=trade.take_profit,
                            confidence=signal['confidence'],
                            reason=signal.get('analysis', {}).get('reason', 'Test trade')
                        )
            
            # 8. Performance Monitoring
            performance_data = {
                'total_trades': 1,
                'winning_trades': 0,
                'profit': 0.0,
                'win_rate': 0.0
            }
            
            self.telegram_bot.send_performance_update(
                chat_id=self.test_chat_id,
                total_trades=performance_data['total_trades'],
                winning_trades=performance_data['winning_trades'],
                total_profit=performance_data['profit']
            )
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")
        
        finally:
            # Cleanup
            if hasattr(self, 'mt5_handler'):
                self.mt5_handler.shutdown()

    def test_error_handling(self):
        """Test error handling throughout the trading cycle."""
        # Test invalid symbol
        signal = self.signal_generator.generate_signal(
            df=pd.DataFrame(),  # Empty DataFrame
            symbol="INVALID",
            timeframe=self.timeframe,
            mtf_data={}
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal['signal_type'], 'HOLD')

        # Test invalid timeframe
        market_analysis = self.market_analysis.analyze(
            df=pd.DataFrame(),  # Empty DataFrame
            symbol=self.symbol,
            timeframe="INVALID"
        )
        self.assertIsNotNone(market_analysis)
        self.assertIsInstance(market_analysis, dict)

if __name__ == '__main__':
    unittest.main() 