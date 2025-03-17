# Multi-Timeframe (MTF) Analysis Module Documentation

## Overview
The MTF Analysis module provides comprehensive analysis of market data across multiple timeframes. It implements a hierarchical approach to timeframe analysis with weighted influence from higher timeframes on lower timeframes.

## Class: MTFAnalysis

### Constructor
```python
def __init__(self)
```
Initializes the MTFAnalysis class with:
- Timeframe hierarchy (M1 to W1)
- Weighted influence for each timeframe
- Timeframe relationships for trading
- Minimum alignment requirements

### Core Analysis Functions

#### analyze
```python
def analyze(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], timeframe: Optional[str] = None) -> Dict
```
Main analysis function that handles both single and multi-timeframe analysis. Returns comprehensive analysis results including trend, score, and key levels.

#### analyze_mtf
```python
def analyze_mtf(self, dataframes: Dict[str, Union[pd.DataFrame, pd.Series]], current_timeframe: Optional[str] = None) -> Dict
```
Analyzes price action across multiple timeframes with improved hierarchy and weighting. Returns detailed timeframe alignment information.

### Timeframe Management

#### get_higher_timeframes
```python
def get_higher_timeframes(self, timeframe: str) -> List[str]
```
Returns a list of all higher timeframes for a given timeframe based on the hierarchy.

### Analysis Functions

#### _analyze_single_timeframe
```python
def _analyze_single_timeframe(self, df: pd.DataFrame) -> float
```
Analyzes a single timeframe's price action using multiple technical indicators. Returns a score between -1 and 1.

#### analyze_timeframe_correlation
```python
def analyze_timeframe_correlation(self, timeframes_data: Dict[str, pd.DataFrame]) -> Dict
```
Analyzes correlation between different timeframes to identify alignment and divergence.

### Trend Analysis

#### _determine_trend_strength
```python
def _determine_trend_strength(self, score: float) -> str
```
Determines trend strength based on analysis score. Returns 'strong', 'moderate', 'weak', or 'neutral'.

#### _calculate_alignment_scores
```python
def _calculate_alignment_scores(self, timeframe_trends: Dict, current_timeframe: Optional[str]) -> Dict
```
Calculates alignment scores between timeframes with weighted influence.

### Market Structure Analysis

#### _calculate_overall_bias
```python
def _calculate_overall_bias(self, timeframe_trends: Dict) -> Dict
```
Calculates overall market bias with weighted timeframe influence.

#### _check_timeframe_alignment
```python
def _check_timeframe_alignment(self, timeframe_trends: Dict, current_timeframe: Optional[str]) -> Dict
```
Checks if the current timeframe is aligned with higher timeframes.

### Support/Resistance Analysis

#### _find_key_levels
```python
def _find_key_levels(self, df: pd.DataFrame) -> List[float]
```
Identifies key price levels using volume profile and swing points.

#### _aggregate_key_levels
```python
def _aggregate_key_levels(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]
```
Aggregates key price levels from multiple timeframes.

### Correlation Analysis

#### calculate_timeframe_correlation
```python
def calculate_timeframe_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float
```
Calculates correlation between two timeframes using price returns.

## Usage Example
```python
mtf_analyzer = MTFAnalysis()

# Single timeframe analysis
single_result = mtf_analyzer.analyze(df, timeframe="H1")

# Multi-timeframe analysis
timeframes_data = {
    "H1": h1_df,
    "H4": h4_df,
    "D1": d1_df
}
mtf_result = mtf_analyzer.analyze_mtf(timeframes_data, current_timeframe="H1")
```

## Notes
- Implements hierarchical timeframe analysis
- Uses weighted influence from higher timeframes
- Provides comprehensive alignment checking
- Includes correlation analysis between timeframes
- Supports both DataFrame and Series inputs
- Implements error handling and data validation
- Returns detailed analysis with confidence metrics