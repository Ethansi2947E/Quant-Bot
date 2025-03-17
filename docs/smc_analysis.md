# Smart Money Concepts (SMC) Analysis Module Documentation

## Overview
The SMC Analysis module implements advanced market analysis techniques based on Smart Money Concepts. It focuses on detecting institutional trading patterns, liquidity sweeps, manipulation points, and other high-probability trading setups.

## Class: SMCAnalysis

### Constructor
```python
def __init__(self)
```
Initializes the SMCAnalysis class with predefined thresholds for various SMC patterns:
- `equal_level_threshold`: 0.0001 (1 pip) for equal high/low detection
- `liquidity_threshold`: 0.0020 (20 pips) for liquidity pool size
- `manipulation_threshold`: 0.0015 (15 pips) for manipulation moves
- `ob_threshold`: 0.0015 (15 pips) for order block size

### Core Analysis Functions

#### analyze
```python
def analyze(self, df: pd.DataFrame) -> Dict
```
Main analysis function that performs comprehensive SMC pattern detection. Returns a dictionary containing:
- Liquidity sweeps
- Manipulation points
- Breaker blocks
- Mitigation blocks
- Premium/discount zones
- Inefficient price moves
- Order flow analysis

### Pattern Detection Functions

#### _detect_liquidity_sweeps
```python
def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]
```
Detects liquidity sweeps (stop hunts) in price action. Identifies areas where price sweeps beyond a level before reversing.

#### _detect_manipulation
```python
def _detect_manipulation(self, df: pd.DataFrame) -> List[Dict]
```
Detects manipulation points such as stop runs and liquidity grabs. These are points where price makes a fake move in one direction before strongly reversing.

#### _find_breaker_blocks
```python
def _find_breaker_blocks(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Identifies breaker blocks, which are strong reversal points in the market structure. Returns both bullish and bearish breaker blocks.

#### _find_mitigation_blocks
```python
def _find_mitigation_blocks(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Detects mitigation blocks, which are areas of unmitigated price that may act as future support/resistance.

### Market Analysis Functions

#### _detect_premium_discount_zones
```python
def _detect_premium_discount_zones(self, df: pd.DataFrame) -> Dict[str, List[Dict]]
```
Identifies premium and discount zones in the market where price is trading significantly above or below average levels.

#### _find_inefficient_moves
```python
def _find_inefficient_moves(self, df: pd.DataFrame) -> List[Dict]
```
Detects inefficient price movements such as gaps or strong directional moves that may need to be filled or revisited.

#### _analyze_order_flow
```python
def _analyze_order_flow(self, df: pd.DataFrame) -> Dict
```
Analyzes institutional order flow based on volume and price action patterns. Returns order flow bias and strength.

### Utility Functions

#### check_liquidity
```python
def check_liquidity(self, analysis: Dict) -> bool
```
Checks for favorable liquidity conditions based on market analysis data. Used to validate trading opportunities.

## Usage Example
```python
smc_analyzer = SMCAnalysis()
df = pd.DataFrame(...)  # Your OHLCV data
analysis_result = smc_analyzer.analyze(df)
```

## Notes
- All price thresholds are in decimal format (not pips)
- Analysis requires high-quality price data with volume
- Best used in conjunction with traditional technical analysis
- Designed for forex markets but applicable to other markets
- All functions include comprehensive error handling 