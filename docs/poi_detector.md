# Points of Interest (POI) Detector Module Documentation

## Overview
The POI Detector module identifies and analyzes significant price levels and zones in the market, focusing on supply and demand zones, liquidity areas, and institutional order flow.

## Classes

### POI (Data Class)
```python
@dataclass
class POI:
```
Data class representing supply/demand zones with attributes:
- `type`: 'supply' or 'demand'
- `price_start`: Zone start price
- `price_end`: Zone end price
- `time`: Timestamp
- `strength`: Zone strength
- `timeframe`: Analysis timeframe
- `status`: 'active', 'tested', or 'broken'
- `volume_imbalance`: Volume imbalance measure
- `delta`: Buy volume - Sell volume

### POIDetector

#### Constructor
```python
def __init__(self, min_volume_threshold: float = 1.5, min_poi_distance: float = 10)
```
Initializes POIDetector with configurable parameters:
- `min_volume_threshold`: Minimum volume multiplier vs average
- `min_poi_distance`: Minimum distance between POIs in pips

### Core Analysis Functions

#### detect_supply_demand_zones
```python
def detect_supply_demand_zones(self, df: pd.DataFrame, timeframe: str) -> Dict[str, List[POI]]
```
Detects supply and demand zones based on volume and price action. Returns dictionary with supply and demand zone lists.

#### analyze_pois
```python
async def analyze_pois(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict
```
Performs comprehensive POI analysis including:
- Zone detection
- Status updates
- Distance calculations
- Quality assessment

### Zone Detection Functions

#### _detect_supply_zones
```python
def _detect_supply_zones(self, df: pd.DataFrame, timeframe: str) -> List[POI]
```
Detects supply (resistance) zones based on bearish price action and volume.

#### _detect_demand_zones
```python
def _detect_demand_zones(self, df: pd.DataFrame, timeframe: str) -> List[POI]
```
Detects demand (support) zones based on bullish price action and volume.

### Analysis Functions

#### _calculate_zone_strength
```python
def _calculate_zone_strength(self, df: pd.DataFrame, index: int, zone_type: str, zone_high: float, zone_low: float, volume_ratio: float) -> float
```
Calculates the strength of a POI zone based on multiple factors.

#### check_poi_alignment
```python
def check_poi_alignment(self, current_analysis: Dict, higher_tf_analysis: Dict, current_price: float, atr: Optional[float] = None) -> bool
```
Checks if POIs are aligned across timeframes for stronger confirmation.

### Data Processing Functions

#### _prepare_data
```python
def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame
```
Prepares DataFrame with required indicators and metrics for POI detection.

#### _update_poi_status
```python
def _update_poi_status(self, poi_zones: Dict[str, List[POI]], current_price: float)
```
Updates POI status based on current price interaction.

### Filtering Functions

#### _filter_overlapping_zones
```python
def _filter_overlapping_zones(self, zones: List[POI]) -> List[POI]
```
Filters out overlapping POI zones, keeping the strongest ones.

#### _zones_overlap
```python
def _zones_overlap(self, zone1: POI, zone2: POI) -> bool
```
Checks if two POI zones overlap in price range.

### Result Processing

#### _process_poi_results
```python
def _process_poi_results(self, poi_zones: Dict[str, List[POI]], current_price: float) -> Dict
```
Processes POI detection results and returns comprehensive analysis.

## Usage Example
```python
detector = POIDetector()
df = pd.DataFrame(...)  # Your OHLCV data

# Detect zones
zones = detector.detect_supply_demand_zones(df, timeframe="H1")

# Full analysis
analysis = await detector.analyze_pois(df, symbol="EURUSD", timeframe="H1")
```

## Notes
- Requires OHLCV data with volume information
- Implements advanced filtering to reduce false signals
- Provides real-time status updates for zones
- Supports multi-timeframe alignment checking
- Includes comprehensive error handling
- Returns detailed analysis with multiple metrics
- Uses volume profile for zone strength calculation 