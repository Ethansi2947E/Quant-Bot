import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger
from dataclasses import dataclass
from datetime import datetime

@dataclass
class POI:
    type: str  # 'supply' or 'demand'
    price_start: float
    price_end: float
    time: datetime
    strength: float
    timeframe: str
    status: str  # 'active', 'tested', 'broken'
    volume_imbalance: float
    delta: float  # Buy volume - Sell volume

class POIDetector:
    def __init__(self):
        self.min_volume_threshold = 1.5  # Minimum volume multiplier vs average
        self.min_poi_distance = 10  # Minimum distance between POIs in pips
        
    def detect_supply_demand_zones(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> Dict[str, List[POI]]:
        """Detect supply and demand zones based on volume and price action."""
        try:
            supply_zones = []
            demand_zones = []
            
            # Calculate volume metrics
            df['volume_ma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            
            # Calculate price deltas
            df['price_delta'] = df['close'] - df['open']
            df['high_delta'] = df['high'] - df['close']
            df['low_delta'] = df['open'] - df['low']
            
            # Detect supply zones (resistance)
            for i in range(20, len(df)-1):
                # Check for high volume bearish candle
                if (df['volume_ratio'].iloc[i] > self.min_volume_threshold and 
                    df['price_delta'].iloc[i] < 0):
                    
                    # Calculate zone boundaries
                    zone_high = max(df['high'].iloc[i], df['high'].iloc[i+1])
                    zone_low = min(df['low'].iloc[i], df['low'].iloc[i+1])
                    
                    # Calculate zone strength
                    strength = self._calculate_zone_strength(
                        df, i, 'supply',
                        zone_high, zone_low,
                        df['volume_ratio'].iloc[i]
                    )
                    
                    # Create supply POI
                    poi = POI(
                        type='supply',
                        price_start=zone_high,
                        price_end=zone_low,
                        time=df.index[i],
                        strength=strength,
                        timeframe=timeframe,
                        status='active',
                        volume_imbalance=df['volume_ratio'].iloc[i],
                        delta=df['price_delta'].iloc[i]
                    )
                    supply_zones.append(poi)
            
            # Detect demand zones (support)
            for i in range(20, len(df)-1):
                # Check for high volume bullish candle
                if (df['volume_ratio'].iloc[i] > self.min_volume_threshold and 
                    df['price_delta'].iloc[i] > 0):
                    
                    # Calculate zone boundaries
                    zone_high = max(df['high'].iloc[i], df['high'].iloc[i+1])
                    zone_low = min(df['low'].iloc[i], df['low'].iloc[i+1])
                    
                    # Calculate zone strength
                    strength = self._calculate_zone_strength(
                        df, i, 'demand',
                        zone_high, zone_low,
                        df['volume_ratio'].iloc[i]
                    )
                    
                    # Create demand POI
                    poi = POI(
                        type='demand',
                        price_start=zone_high,
                        price_end=zone_low,
                        time=df.index[i],
                        strength=strength,
                        timeframe=timeframe,
                        status='active',
                        volume_imbalance=df['volume_ratio'].iloc[i],
                        delta=df['price_delta'].iloc[i]
                    )
                    demand_zones.append(poi)
            
            # Filter overlapping zones
            supply_zones = self._filter_overlapping_zones(supply_zones)
            demand_zones = self._filter_overlapping_zones(demand_zones)
            
            return {
                'supply': supply_zones,
                'demand': demand_zones
            }
            
        except Exception as e:
            logger.error(f"Error detecting POIs: {str(e)}")
            return {'supply': [], 'demand': []}
    
    def _calculate_zone_strength(
        self,
        df: pd.DataFrame,
        index: int,
        zone_type: str,
        zone_high: float,
        zone_low: float,
        volume_ratio: float
    ) -> float:
        """Calculate the strength of a POI zone."""
        try:
            # Base strength from volume
            strength = min(volume_ratio / 2, 1.0)
            
            # Add strength based on price rejection
            if zone_type == 'supply':
                rejection = df['high_delta'].iloc[index] / (zone_high - zone_low)
                strength += min(rejection * 0.3, 0.3)
            else:
                rejection = df['low_delta'].iloc[index] / (zone_high - zone_low)
                strength += min(rejection * 0.3, 0.3)
            
            # Add strength based on subsequent price respect
            respect_count = 0
            for i in range(index + 1, min(index + 10, len(df))):
                if zone_type == 'supply' and df['high'].iloc[i] <= zone_high:
                    respect_count += 1
                elif zone_type == 'demand' and df['low'].iloc[i] >= zone_low:
                    respect_count += 1
                    
            strength += min(respect_count * 0.05, 0.2)
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating zone strength: {str(e)}")
            return 0.5
    
    def _filter_overlapping_zones(self, zones: List[POI]) -> List[POI]:
        """Filter out overlapping POI zones, keeping the strongest ones."""
        if not zones:
            return zones
            
        # Sort zones by strength
        zones.sort(key=lambda x: x.strength, reverse=True)
        
        filtered_zones = []
        for zone in zones:
            # Check if this zone overlaps with any stronger zones
            overlapping = False
            for filtered_zone in filtered_zones:
                if self._zones_overlap(zone, filtered_zone):
                    overlapping = True
                    break
                    
            if not overlapping:
                filtered_zones.append(zone)
                
        return filtered_zones
    
    def _zones_overlap(self, zone1: POI, zone2: POI) -> bool:
        """Check if two POI zones overlap."""
        return not (zone1.price_end > zone2.price_start or 
                   zone2.price_end > zone1.price_start) 