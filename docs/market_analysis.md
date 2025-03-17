# Market Analysis Module Documentation

## Overview
The Market Analysis module provides comprehensive tools for technical analysis, market structure detection, and trading pattern identification. This module is designed to analyze financial market data and identify various trading opportunities based on technical indicators and price action patterns.

## Class: MarketAnalysis

### Constructor
```python
def __init__(self, ob_threshold: float = 0.0015)
```
Initializes the MarketAnalysis class with a configurable order block threshold.

### Core Analysis Functions

#### analyze_market
```python
async def analyze_market(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]
```
Main analysis function that performs comprehensive market analysis including trend detection, structure analysis, and pattern identification for a given symbol and timeframe.

#### detect_market_structure
```python
def detect_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]
```
Analyzes and detects the overall market structure including swing points, structure breaks, and market patterns.

### Technical Indicators

#### calculate_atr
```python
def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series
```
Calculates the Average True Range (ATR) indicator for volatility measurement.

#### calculate_rsi
```python
def calculate_rsi(self, prices, period=14)
```
Computes the Relative Strength Index (RSI) for momentum analysis.

### Pattern Detection

#### detect_swing_points
```python
def detect_swing_points(self, df: pd.DataFrame, swing_size: int = None) -> Dict[str, List[Dict[str, Any]]]
```
Identifies swing high and low points in the price action.

#### detect_order_blocks
```python
def detect_order_blocks(self, df: pd.DataFrame, swing_points: Tuple[List[Dict], List[Dict]]) -> Dict[str, List[Dict]]
```
Identifies potential order blocks based on swing points and price action.

#### detect_fair_value_gaps
```python
def detect_fair_value_gaps(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Detects fair value gaps in price action.

#### detect_bos_and_choch
```python
def detect_bos_and_choch(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]], confirmation_type: str = 'Candle Close') -> Dict[str, List[Dict[str, Any]]]
```
Identifies Break of Structure (BOS) and Change of Character (CHoCH) patterns.

### Market State Analysis

#### analyze_trend
```python
def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]
```
Analyzes the current market trend and provides trend characteristics.

#### classify_volatility
```python
def classify_volatility(self, atr: pd.Series) -> str
```
Classifies market volatility based on ATR values.

#### detect_killzones
```python
def detect_killzones(self, df: pd.DataFrame) -> Dict[str, Any]
```
Identifies high-probability trading zones based on market session analysis.

### Structure Analysis

#### _assess_structure_quality
```python
def _assess_structure_quality(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> Dict
```
Evaluates the quality of market structure based on swing points.

#### detect_structure_breaks
```python
def detect_structure_breaks(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]
```
Identifies points where market structure breaks occur.

### Market Session Analysis

#### get_current_session
```python
def get_current_session(self) -> Tuple[str, Dict]
```
Determines the current market session (Asian, London, New York).

#### detect_displacement
```python
def detect_displacement(self, df: pd.DataFrame) -> Dict[str, bool]
```
Analyzes price displacement between different market sessions.

### Helper Functions

#### _calculate_momentum
```python
def _calculate_momentum(self, df: pd.DataFrame) -> float
```
Calculates market momentum using various technical indicators.

#### _calculate_trend_consistency
```python
def _calculate_trend_consistency(self, df: pd.DataFrame) -> float
```
Measures the consistency of the current market trend.

#### timeinrange
```python
def timeinrange(self, current_time: time, start_time: time, end_time: time) -> bool
```
Utility function to check if a given time falls within a specified range.

#### _calculate_market_quality
```python
def _calculate_market_quality(self, trend_strength: float, momentum: float, volume_trend: float, volatility_state: str, market_state: str) -> float
```
Computes overall market quality score based on multiple factors.

### Volume Analysis

#### _calculate_volume_trend
```python
def _calculate_volume_trend(self, df: pd.DataFrame) -> Dict
```
Analyzes volume trends and their relationship with price action.

### Smart Money Concepts (SMC)

#### detect_mss_bullish
```python
def detect_mss_bullish(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]
```
Detects bullish market structure shifts using SMC concepts.

#### detect_mss_bearish
```python
def detect_mss_bearish(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]
```
Detects bearish market structure shifts using SMC concepts.

## Usage Example
```python
analyzer = MarketAnalysis()
market_data = await analyzer.get_market_data("BTCUSDT", "1h")
analysis_result = await analyzer.analyze_market("BTCUSDT", "1h")
```

## Notes
- All time-based functions use UTC timezone
- Price data should be in OHLCV format
- Minimum data requirements vary by analysis function
- Error handling is implemented throughout the module 