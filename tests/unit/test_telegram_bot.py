import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from src.telegram_bot import TelegramBot
from src.models import Trade

@pytest.mark.asyncio
class TestTelegramBot:
    @pytest.fixture
    def bot(self, monkeypatch):
        """Setup test bot instance."""
        # Create bot instance
        bot = TelegramBot()
        
        # Mock the telegram bot instance
        mock_bot = AsyncMock()
        mock_application = AsyncMock()
        
        # Set the mocks
        bot.bot = mock_bot
        bot.application = mock_application
        
        # Set allowed user IDs for testing
        bot.allowed_user_ids = ["123456789"]
        
        # Mock TELEGRAM_CONFIG
        mock_config = {
            "bot_token": "test_token",
            "allowed_user_ids": ["123456789"]
        }
        monkeypatch.setattr("src.telegram_bot.TELEGRAM_CONFIG", mock_config)
        
        return bot

    async def test_authentication(self, bot):
        """Test user authentication for authorized and unauthorized users."""
        # Test authorized user
        assert await bot.check_auth(123456789) is True
        # Test unauthorized user
        assert await bot.check_auth(987654321) is False

    async def test_message_formatting(self, bot):
        """Test message formatting functions for trade alerts."""
        message = bot.format_alert(
            symbol="EURUSD",
            direction="BUY",
            entry=1.1000,
            sl=1.0950,
            tp=1.1100,
            confidence=0.85,
            reason="Strong uptrend on H4"
        )
        
        assert "EURUSD" in message
        assert "BUY" in message
        assert "1.1000" in message
        assert "1.0950" in message
        assert "1.1100" in message
        assert "85%" in message
        assert "Strong uptrend on H4" in message

    async def test_send_trade_alert(self, bot):
        """Test sending trade alerts to Telegram."""
        await bot.send_trade_alert(
            chat_id=123456789,
            symbol="EURUSD",
            direction="BUY",
            entry=1.1000,
            sl=1.0950,
            tp=1.1100,
            confidence=0.85,
            reason="Strong uptrend on H4"
        )
        
        bot.bot.send_message.assert_called_once()
        args = bot.bot.send_message.call_args[1]
        assert args["chat_id"] == 123456789
        assert "EURUSD" in args["text"]
        assert "BUY" in args["text"]

    async def test_send_error_notification(self, bot):
        """Test sending error notifications to Telegram."""
        error_message = "Test error message"
        await bot.notify_error(chat_id=123456789, error=error_message)
        
        bot.bot.send_message.assert_called_once()
        args = bot.bot.send_message.call_args[1]
        assert args["chat_id"] == 123456789
        assert error_message in args["text"]

    async def test_send_performance_update(self, bot):
        """Test sending performance updates to Telegram."""
        await bot.send_performance_update(
            chat_id=123456789,
            total_trades=10,
            winning_trades=7,
            total_profit=500.50
        )
        
        bot.bot.send_message.assert_called_once()
        args = bot.bot.send_message.call_args[1]
        assert args["chat_id"] == 123456789
        assert "70%" in args["text"]  # Win rate
        assert "500.50" in args["text"]  # Total profit

    async def test_command_handling(self, bot):
        """Test handling of bot commands."""
        message = AsyncMock()
        message.chat.id = 123456789
        message.text = "/start"
        
        await bot.process_command(message)
        
        bot.bot.send_message.assert_called_once()
        args = bot.bot.send_message.call_args[1]
        assert args["chat_id"] == 123456789
        assert "Welcome" in args["text"]

if __name__ == '__main__':
    pytest.main() 