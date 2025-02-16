# Market Analysis Documentation

## Overview
The market analysis module provides comprehensive analysis of market structure, trends, and conditions using multiple components and advanced filtering techniques.

## Components

### 1. Swing Point Detection
```python
def detect_swing_points(df: pd.DataFrame, lookback: int = None) -> Tuple[List[Dict], List[Dict]]
```

#### Features
- Dynamic lookback period based on ATR volatility
- Local volatility-based threshold adjustment
- Strength metrics for swing points
- Minimum lookback protection (5 periods)

#### Implementation Details
- Calculates Average True Range (ATR) for volatility measurement
- Adjusts lookback period: `lookback = max(swing_detection_lookback * (1 + avg_atr * 10), 5)`
- Uses local price range for threshold adjustment
- Validates swing point significance using size comparison
- Returns both swing highs and lows with strength metrics

### 2. Market Bias Detection
```python
def _determine_market_bias(swing_highs: List[Dict], swing_lows: List[Dict], structure_breaks: List[Dict]) -> str
```

#### Weighted Scoring System
1. Swing Points Analysis (40% weight)
   - Analyzes last 4 swing points
   - Counts higher highs/lows and lower highs/lows
   - Score contribution: 20% each for highs and lows

2. Structure Breaks Analysis (30% weight)
   - Examines last 5 structure breaks
   - Applies decreasing weights for older breaks
   - Weight calculation: `0.3 * (1 - i * 0.15)`

3. Swing Point Quality (30% weight)
   - Considers average swing sizes
   - Compares with minimum size threshold
   - Adds directional bias based on progression

#### Bias Classifications
- Strong Bullish: Score >= 0.7
- Moderate Bullish: Score >= 0.4
- Strong Bearish: Score <= -0.7
- Moderate Bearish: Score <= -0.4
- Neutral: All other cases

### 3. Structure Quality Assessment
```python
def _assess_structure_quality(swing_highs: List[Dict], swing_lows: List[Dict], order_blocks: List[Dict], fvgs: List[Dict]) -> float
```

#### Quality Metrics
1. Swing Point Count (30%)
   - Minimum required points check
   - Equal weight for highs and lows

2. Swing Point Spacing (20%)
   - Average spacing calculation
   - Comparison with lookback period

3. Order Block Presence (20%)
   - Validates significant order blocks
   - Considers both bullish and bearish

4. Fair Value Gaps (20%)
   - Checks for valid FVGs
   - Both bullish and bearish gaps

5. Swing Point Size (10%)
   - Average size calculation
   - Minimum size threshold check

### 4. Market State Classification
```python
def classify_trend(df: pd.DataFrame) -> str
```

#### Trend States
- Strong Uptrend: All EMAs aligned, strong momentum
- Uptrend: EMAs aligned, positive slope
- Strong Downtrend: Reverse EMA alignment, negative momentum
- Downtrend: Reverse EMA alignment, negative slope
- Ranging: Minimal EMA slopes
- Choppy: Default state when unclear

#### Implementation
- Uses EMA20, EMA50, and EMA200
- Calculates price position relative to EMAs
- Measures EMA slopes and momentum
- Dynamic threshold adjustments

### 5. Volatility Classification
```python
def classify_volatility(atr: pd.Series) -> str
```

#### States
- Extreme: > mean + 2*std
- High: > mean + std
- Normal: Within ±std
- Low: < mean - std

#### Features
- Dynamic thresholds based on recent ATR
- 20-period rolling window
- Percentile-based classification
- Standard deviation bands

## Usage Example

```python
market_analysis = MarketAnalysis()

# Analyze market structure
structure = market_analysis.analyze_market_structure(df, symbol, timeframe)

# Get market conditions
volatility_state = market_analysis.classify_volatility(df['atr'])
trend_state = market_analysis.classify_trend(df)

# Check trading suitability
is_suitable = market_analysis._is_suitable_for_trading(
    market_state=trend_state,
    volatility_state=volatility_state,
    trend_strength=structure['structure_quality'],
    session=current_session,
    timeframe=timeframe,
    market_quality=structure['structure_quality']
)
```

## Configuration

### Thresholds
- `swing_detection_lookback`: Default 15 periods
- `swing_detection_threshold`: 0.0008
- `min_swing_size`: 0.0020
- `ob_threshold`: 0.0015
- `structure_break_threshold`: 0.0020
- `min_swing_points`: 4

### Timeframe-Specific Settings
Available in the `MARKET_STRUCTURE_CONFIG` dictionary:
- Lookback periods
- Threshold values
- Minimum swing sizes
- Order block parameters
- Structure break requirements

## Error Handling
- Comprehensive error catching and logging
- Fallback to safe default values
- Detailed error messages for debugging
- Graceful degradation of functionality 