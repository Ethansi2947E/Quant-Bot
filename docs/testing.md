# Testing Framework Documentation

## Overview
The `tests` folder contains comprehensive unit tests, integration tests, and backtesting validation suites to ensure the reliability and accuracy of the trading system.

## Test Structure

```
tests/
├── unit/
│   ├── test_market_analysis.py
│   ├── test_signal_generator.py
│   ├── test_smc_analysis.py
│   ├── test_mtf_analysis.py
│   └── test_risk_manager.py
├── integration/
│   ├── test_trading_flow.py
│   ├── test_mt5_integration.py
│   └── test_telegram_integration.py
└── backtest/
    ├── test_historical_data.py
    ├── test_strategy_validation.py
    └── test_performance_metrics.py
```

## Unit Tests

### Market Analysis Tests
- Swing point detection accuracy
- Order block identification
- Fair value gap detection
- Structure break analysis
- Market bias calculation

### Signal Generator Tests
- Score calculation accuracy
- Component weight application
- Threshold validation
- Signal classification
- Risk parameter calculation

### Smart Money Concepts Tests
- Liquidity sweep detection
- Manipulation point identification
- Order block validation
- Breaker block detection
- Mitigation block analysis

### Multi-Timeframe Tests
- Timeframe alignment scoring
- Trend correlation analysis
- Momentum confirmation
- Structure alignment validation

### Risk Manager Tests
- Position size calculation
- Stop loss placement
- Risk-reward ratio validation
- Maximum exposure checks
- Drawdown protection

## Integration Tests

### Trading Flow Tests
- End-to-end trade execution
- Signal to order workflow
- Position management
- Risk management integration
- Performance tracking

### MT5 Integration Tests
- Connection management
- Data retrieval accuracy
- Order execution reliability
- Position tracking
- Error handling

### Telegram Integration Tests
- Command processing
- Alert generation
- Status updates
- Error notifications
- User authentication

## Backtest Validation

### Historical Data Tests
- Data integrity checks
- Price data validation
- Volume data accuracy
- Timeframe consistency
- Gap detection

### Strategy Validation Tests
- Entry/exit accuracy
- Signal generation timing
- Risk management compliance
- Performance metric calculation
- Edge case handling

### Performance Metrics Tests
- Profit calculation accuracy
- Drawdown measurement
- Risk-adjusted returns
- Win rate calculation
- Trade statistics

## Running Tests

### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/unit/

# Run specific test file
python -m pytest tests/unit/test_market_analysis.py

# Run with coverage report
python -m pytest tests/unit/ --cov=src/
```

### Integration Tests
```bash
# Run all integration tests
python -m pytest tests/integration/

# Run specific integration test
python -m pytest tests/integration/test_trading_flow.py
```

### Backtest Validation
```bash
# Run all backtest validation tests
python -m pytest tests/backtest/

# Run with detailed output
python -m pytest tests/backtest/ -v
```

## Test Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    backtest: Backtest validation tests
```

### conftest.py
Contains shared fixtures and configuration for tests:
- MT5 mock data
- Sample price data
- Test configurations
- Common utilities
- Mock objects

## Best Practices

### Writing Tests
1. Follow AAA pattern (Arrange, Act, Assert)
2. Use meaningful test names
3. Test edge cases
4. Mock external dependencies
5. Keep tests independent

### Test Coverage
- Aim for >80% code coverage
- Focus on critical components
- Include edge cases
- Test error handling
- Validate business logic

### Continuous Integration
- Tests run on every commit
- Coverage reports generated
- Performance benchmarks
- Integration test suite
- Regression testing

## Maintenance

### Regular Tasks
1. Update test data
2. Review coverage reports
3. Add tests for new features
4. Maintain mock objects
5. Update documentation

### Performance
- Monitor test execution time
- Optimize slow tests
- Use appropriate fixtures
- Clean up test data
- Profile test suite

## Error Handling

### Common Issues
1. MT5 connection failures
2. Data inconsistencies
3. Timing issues
4. Resource cleanup
5. Mock data limitations

### Solutions
- Robust error handling
- Detailed logging
- Clean test isolation
- Resource management
- Mock data validation 