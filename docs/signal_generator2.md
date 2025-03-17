# Signal Generator Implementation Guide

## Overview
This document details the implementation of a comprehensive trading signal generator that integrates multiple analysis modules to execute the **BMS + SH + RTO + Turtle Soup** strategy. The system combines market structure analysis, liquidity detection, and multi-timeframe confirmation with strict risk management.

![Strategy Workflow](https://via.placeholder.com/800x400.png?text=Strategy+Workflow+Diagram)

## Module Integration Matrix

| Module               | Key Functions Used                          | Strategy Component Served       |
|----------------------|---------------------------------------------|----------------------------------|
| `MarketAnalysis`     | `detect_market_structure()`<br>`get_current_session()`<br>`detect_order_blocks()` | BMS Detection<br>Session Timing<br>RTO Levels |
| `SMCAnalysis`        | `analyze()`<br>`_detect_liquidity_sweeps()` | Stop Hunt Identification<br>Liquidity Zones |
| `POIDetector`        | `detect_supply_demand_zones()`<br>`analyze_pois()` | Order Block Detection<br>Zone Strength Analysis |
| `MTFAnalysis`        | `analyze_mtf()`<br>`check_timeframe_alignment()` | Trend Confirmation<br>HTF-LTF Alignment |
| `DivergenceAnalysis` | `analyze()`                                 | Pattern Confirmation            |
| `RiskManager`        | `calculate_position_size()`<br>`validate_trade()` | Position Sizing<br>Trade Validation |
| `MT5Handler`         | `get_rates()`<br>`place_market_order()`     | Data Fetching<br>Trade Execution |
| `TelegramBot`        | `send_trade_alert()`                        | Signal Notifications            |

## Core Workflow

### 1. Data Acquisition & Preprocessing
```python
# Using MT5Handler
df = await self.mt5.get_rates("EURUSD", "M15", 1000)
mtf_data = {
    "H1": await self.mt5.get_rates("EURUSD", "H1", 500),
    "H4": await self.mt5.get_rates("EURUSD", "H4", 500),
    "D1": await self.mt5.get_rates("EURUSD", "D1", 500)
}
2. Market Structure Analysis
python
Copy
# Using MarketAnalysis
market_structure = self.market_analyzer.detect_market_structure(df)
swing_points = self.market_analyzer.detect_swing_points(df)
order_blocks = self.market_analyzer.detect_order_blocks(df, swing_points)
3. Liquidity & SMC Analysis
python
Copy
# Using SMCAnalysis and POIDetector
smc_signals = self.smc_analyzer.analyze(df)
poi_analysis = await self.poi_detector.analyze_pois(df, "EURUSD", "M15")

# Detect key levels
supply_zones = poi_analysis['supply_zones']
demand_zones = poi_analysis['demand_zones']
4. Multi-Timeframe Confirmation
python
Copy
# Using MTFAnalysis
mtf_analysis = self.mtf_analyzer.analyze_mtf(mtf_data, "M15")
htf_bias = mtf_analysis['D1']['bias']  # Daily timeframe bias
5. Strategy Pattern Matching
Turtle Soup Detection
python
Copy
if self._is_london_session() and htf_bias == 'bearish':
    for sweep in smc_signals['liquidity_sweeps']:
        if sweep['type'] == 'BSL' and sweep['strength'] > 0.8:
            entry = current_price
            sl = sweep['level'] + 0.0020
            tp = self._calculate_fib_extension(sweep)
SH+BMS+RTO Detection
python
Copy
for structure_break in market_structure['breaks']:
    if structure_break['type'] == 'BMS' and self._valid_rto(structure_break):
        nearest_ob = self._find_nearest_ob(order_blocks, structure_break['price'])
        entry = nearest_ob['price']
        sl = nearest_ob['sl_level']
        tp = self._calculate_liquidity_target(structure_break)
6. Risk Validation
python
Copy
validated = []
for signal in signals:
    size = self.risk_mgr.calculate_position_size(
        account_balance=account_info['balance'],
        entry_price=signal['entry'],
        stop_loss_price=signal['sl'],
        symbol=symbol
    )
    if self.risk_mgr.validate_trade(size, ...):
        validated.append(signal)
7. Execution & Alerting
python
Copy
await self.tg_bot.send_trade_alert(
    chat_id=ADMIN_CHAT_ID,
    symbol="EURUSD",
    direction=signal['direction'],
    entry=1.1200,
    sl=1.1150,
    tp=1.1300,
    confidence=0.85,
    reason="SH+BMS+RTO Confirmed"
)
Key Implementation Details
Market Structure Break Detection
python
Copy
def _detect_bms(self, df):
    """Using MarketAnalysis module"""
    structure = self.market_analyzer.detect_market_structure(df)
    return [
        break_point for break_point in structure['breaks']
        if break_point['confidence'] > 0.7
        and break_point['type'] == 'BMS'
    ]
Order Block Validation
python
Copy
def _validate_ob(self, ob):
    """Using POIDetector and MarketAnalysis"""
    return (
        ob['strength'] > 0.65
        and ob['status'] == 'active'
        and self.poi_detector.check_poi_alignment(
            current_analysis=ob,
            higher_tf_analysis=mtf_analysis
        )
    )
Session-Based Filtering
python
Copy
def _session_filter(self, signal):
    """Using MarketAnalysis session detection"""
    session = self.market_analyzer.get_current_session()
    if signal['type'] == 'TurtleSoup' and session != 'London':
        return False
    if signal['type'] == 'AMD' and session != 'New York':
        return False
    return True
Risk Management Implementation
Dynamic Position Sizing
python
Copy
position_size = self.risk_mgr.calculate_position_size(
    account_balance=10000,
    risk_per_trade=0.01,
    entry_price=1.1200,
    stop_loss_price=1.1150,
    symbol="EURUSD",
    market_condition=mtf_analysis['market_state'],
    volatility_state=market_structure['volatility']
)
Trade Validation Matrix
Condition	Validation Function	Threshold
Daily Risk Exposure	check_daily_limits()	< 2%
Trade Spacing	validate_trade_spacing()	> 4 hours
Consecutive Losses	check_loss_streak()	< 3 trades
Correlation Risk	check_correlation()	< 0.7
Alert Structure Example
json
Copy
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "strategy": "SH+BMS+RTO",
  "entry": 1.1200,
  "sl": 1.1150,
  "tp": 1.1300,
  "confidence": 0.85,
  "session": "London",
  "htf_alignment": {
    "H4": "bullish",
    "D1": "neutral"
  },
  "risk_metrics": {
    "rr_ratio": 1:3,
    "position_size": 0.15
  }
}
Performance Monitoring
python
Copy
await self.tg_bot.send_performance_update(
    chat_id=ADMIN_CHAT_ID,
    total_trades=42,
    winning_trades=32,
    total_profit=1500.00
)