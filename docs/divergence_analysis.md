# Divergence Analysis Module Documentation

## Overview
The Divergence Analysis module provides advanced detection and analysis of various types of divergences in price action and technical indicators. It supports regular, hidden, structural, and momentum divergences with comprehensive validation and quality assessment.

## Class: DivergenceAnalysis

### Constructor
```python
def __init__(self)
```
Initializes the DivergenceAnalysis class with configurable parameters:
- `lookback_period`: 30 periods for analysis window
- `divergence_threshold`: 0.005 (0.5%) for minimum divergence
- `hidden_divergence_threshold`: 0.2
- `momentum_divergence_threshold`: 0.2
- `min_swing_size`: 0.0003 for price movements
- `min_rsi_swing`: 5.0 for RSI movements
- `confirmation_bars`: 2 for pattern confirmation
- `min_data_points`: 50 required for analysis

### Core Analysis Functions

#### analyze
```python
def analyze(self, df: pd.DataFrame) -> Dict
```
Main analysis function that detects and validates multiple types of divergences:
- Regular divergences
- Hidden divergences
- Structural divergences
- Momentum divergences

Returns a dictionary containing all divergence types and their relationships.

### Divergence Detection Functions

#### _find_regular_divergences
```python
def _find_regular_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Detects regular (classic) divergences between price and indicators. Returns both bullish and bearish divergences.

#### _find_hidden_divergences
```python
def _find_hidden_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Detects hidden divergences that indicate trend continuation. Returns both bullish and bearish divergences.

#### _find_structural_divergences
```python
def _find_structural_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Detects structural divergences between swing points. Returns both bullish and bearish divergences.

#### _find_momentum_divergences
```python
def _find_momentum_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Detects momentum-based divergences using multiple indicators. Returns both bullish and bearish divergences.

### Validation Functions

#### _validate_bullish_divergence
```python
def _validate_bullish_divergence(self, window: pd.DataFrame) -> bool
```
Validates bullish divergence patterns with improved sensitivity and confirmation requirements.

#### _validate_bearish_divergence
```python
def _validate_bearish_divergence(self, window: pd.DataFrame) -> bool
```
Validates bearish divergence patterns with improved sensitivity and confirmation requirements.

### Technical Indicators

#### _calculate_indicators
```python
def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame
```
Calculates technical indicators required for divergence analysis:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- OBV (On Balance Volume)
- MFI (Money Flow Index)

### Quality Assessment

#### _assess_indicator_quality
```python
def _assess_indicator_quality(self, df: pd.DataFrame) -> Dict
```
Assesses the quality and reliability of calculated indicators. Returns quality score and issues found.

### Relationship Analysis

#### _analyze_divergence_relationships
```python
def _analyze_divergence_relationships(self, regular: Dict, hidden: Dict, structural: Dict, momentum: Dict) -> Dict
```
Analyzes relationships between different types of divergences for stronger confirmation.

## Usage Example
```python
analyzer = DivergenceAnalysis()
df = pd.DataFrame(...)  # Your OHLCV data
analysis_result = analyzer.analyze(df)
```

## Notes
- Requires high-quality price data with volume
- Includes multiple validation steps for reliability
- Supports various timeframes
- Implements comprehensive error handling
- Uses advanced filtering to reduce false signals
- Provides quality assessment for indicator reliability 