import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from dataclasses import dataclass
from datetime import datetime

@dataclass
class POI:
    """Point of Interest data class representing supply/demand zones."""
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
    """Detector for supply and demand zones based on volume and price action."""
    
    def __init__(self, 
                 min_volume_threshold: float = 1.5,
                 min_poi_distance: float = 10):
        """Initialize POIDetector with configurable parameters.
        
        Args:
            min_volume_threshold: Minimum volume multiplier vs average
            min_poi_distance: Minimum distance between POIs in pips
        """
        self.min_volume_threshold = min_volume_threshold
        self.min_poi_distance = min_poi_distance

    def detect_supply_demand_zones(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> Dict[str, List[POI]]:
        """Detect supply and demand zones based on volume and price action.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data
            
        Returns:
            Dict containing supply and demand zones
        """
        try:
            # Check for minimum required data
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns for POI detection: {missing_columns}")
                return {'supply': [], 'demand': []}
                
            if len(df) < 20:  # Need at least 20 candles for meaningful analysis
                logger.warning(f"Insufficient data for POI detection: {len(df)} candles")
                return {'supply': [], 'demand': []}
            
            logger.debug(f"Starting POI detection with {len(df)} candles on {timeframe}")
            
            # Prepare data
            df = self._prepare_data(df)
            
            # Detect zones
            logger.debug("Detecting supply zones...")
            supply_zones = self._detect_supply_zones(df, timeframe)
            logger.debug(f"Found {len(supply_zones)} initial supply zones")
            
            logger.debug("Detecting demand zones...")
            demand_zones = self._detect_demand_zones(df, timeframe)
            logger.debug(f"Found {len(demand_zones)} initial demand zones")
            
            # Filter overlapping zones
            supply_zones = self._filter_overlapping_zones(supply_zones)
            demand_zones = self._filter_overlapping_zones(demand_zones)
            
            logger.debug(f"After filtering: {len(supply_zones)} supply zones, {len(demand_zones)} demand zones")
            
            return {
                'supply': supply_zones,
                'demand': demand_zones
            }
            
        except Exception as e:
            logger.error(f"Error detecting POIs: {str(e)}")
            logger.exception("Detailed POI detection error:")
            return {'supply': [], 'demand': []}
            
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with required indicators."""
        df = df.copy()
        
        # Calculate volume metrics
        logger.debug(f"Preparing POI data with columns: {list(df.columns)}")
        
        # Check if tick_volume exists, otherwise use volume or create synthetic volume
        if 'tick_volume' in df.columns:
            logger.debug("Using tick_volume from data")
            volume_col = 'tick_volume'
        elif 'volume' in df.columns:
            logger.debug("Using volume instead of tick_volume")
            # Rename volume to tick_volume for consistency
            df['tick_volume'] = df['volume']
            volume_col = 'tick_volume'
        else:
            logger.warning("No volume data found, creating synthetic volume based on price action")
            # Create synthetic volume based on candle range
            df['tick_volume'] = (df['high'] - df['low']) * 1000  # Simple proxy for volume
            volume_col = 'tick_volume'
            logger.debug(f"Created synthetic volume with mean: {df['tick_volume'].mean():.2f}")
        
        try:
            # Calculate volume moving average and ratio
            df['volume_ma'] = df[volume_col].rolling(20).mean()
            df['volume_ratio'] = df[volume_col] / df['volume_ma']
            logger.debug(f"Volume metrics calculated. Mean ratio: {df['volume_ratio'].dropna().mean():.2f}")
            
            # Fill NaN values in volume metrics
            df['volume_ma'] = df['volume_ma'].fillna(df[volume_col].mean())
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            # Calculate price deltas
            df['price_delta'] = df['close'] - df['open']
            df['high_delta'] = df['high'] - df['close']
            df['low_delta'] = df['open'] - df['low']
            
            logger.debug(f"Data preparation successful with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error in POI data preparation: {str(e)}")
            # Return dataframe with minimal required columns
            if 'price_delta' not in df.columns:
                df['price_delta'] = df['close'] - df['open']
            if 'volume_ratio' not in df.columns:
                df['volume_ratio'] = 1.0
            return df
        
    def _detect_supply_zones(self, df: pd.DataFrame, timeframe: str) -> List[POI]:
        """Detect supply (resistance) zones."""
        supply_zones = []
        
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
                
        return supply_zones
        
    def _detect_demand_zones(self, df: pd.DataFrame, timeframe: str) -> List[POI]:
        """Detect demand (support) zones."""
        demand_zones = []
        
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
                
        return demand_zones
    
    def _calculate_zone_strength(
        self,
        df: pd.DataFrame,
        index: int,
        zone_type: str,
        zone_high: float,
        zone_low: float,
        volume_ratio: float
    ) -> float:
        """Calculate the strength of a POI zone with normalized component weighting."""
        try:
            # Define component weights that sum to 1.0
            volume_weight = 0.5       # 50% of strength from volume
            rejection_weight = 0.3    # 30% of strength from price rejection
            respect_weight = 0.2      # 20% of strength from price respect
            
            # Calculate volume component (0-1 scale)
            volume_component = min(volume_ratio / 3, 1.0)  # Normalize to 0-1
            
            # Calculate rejection component (0-1 scale)
            zone_size = max(0.0001, zone_high - zone_low)  # Avoid division by zero
            if zone_type == 'supply':
                rejection = df['high_delta'].iloc[index] / zone_size
            else:  # demand
                rejection = df['low_delta'].iloc[index] / zone_size
            rejection_component = min(rejection, 1.0)  # Cap at 1.0
            
            # Calculate respect component (0-1 scale)
            respect_count = 0
            # Look at subsequent price action to see if the zone is respected
            for i in range(index + 1, min(index + 10, len(df))):
                if zone_type == 'supply' and df['high'].iloc[i] <= zone_high:
                    respect_count += 1
                elif zone_type == 'demand' and df['low'].iloc[i] >= zone_low:
                    respect_count += 1
            respect_component = min(respect_count / 5, 1.0)  # Normalize to 0-1 scale
            
            # Combine components with proper weights
            strength = (
                volume_component * volume_weight +
                rejection_component * rejection_weight +
                respect_component * respect_weight
            )
            
            # Ensure final value is between 0 and 1
            return max(0.0, min(strength, 1.0))
            
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
        # Ensure we're comparing numeric values
        z1_price_start = float(zone1.price_start)
        z1_price_end = float(zone1.price_end)
        z2_price_start = float(zone2.price_start)
        z2_price_end = float(zone2.price_end)
        
        # Fix logic to correctly detect overlap
        # Two zones overlap if one zone doesn't entirely come after or before the other
        return not (z1_price_end > z2_price_start or z2_price_end > z1_price_start)
    
    async def analyze_pois(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Dict:
        """Analyze points of interest with enhanced tracking.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe of the data
            
        Returns:
            Dict containing POI analysis results
        """
        try:
            logger.info(f"Detecting POIs for {symbol} on {timeframe}")
            logger.debug(f"POI Analysis - DataFrame columns: {list(df.columns)}")
            logger.debug(f"POI Analysis - DataFrame shape: {df.shape}")
            
            # Get POIs
            poi_zones = self.detect_supply_demand_zones(df, timeframe)
            
            # Log the results from POI detection
            supply_count = len(poi_zones.get('supply', []))
            demand_count = len(poi_zones.get('demand', []))
            logger.debug(f"POI Detection - Found {supply_count} supply zones and {demand_count} demand zones")
            
            # Get current price for reference
            current_price = float(df['close'].iloc[-1])
            logger.debug(f"POI Detection - Current Price: {current_price}")
            
            # Update POI status based on current price
            self._update_poi_status(poi_zones, current_price)
            
            # Process POI results
            result = self._process_poi_results(poi_zones, current_price)
            
            # Log the final processed results
            logger.debug(f"POI Result Structure: {list(result.keys())}")
            logger.debug(f"POI Result Types - current_price: {type(result.get('current_price'))}, resistance: {type(result.get('resistance'))}, support: {type(result.get('support'))}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_pois: {str(e)}")
            logger.exception("Detailed POI analysis error:")
            # Return empty result
            return self._get_empty_result(df)
    
    def _update_poi_status(self, poi_zones: Dict[str, List[POI]], current_price: float):
        """Update POI status based on current price."""
        for zone_type in ['supply', 'demand']:
            for poi in poi_zones[zone_type]:
                try:
                    price_start = float(poi.price_start)
                    price_end = float(poi.price_end)
                    logger.debug(f"POI {zone_type} zone - Start: {price_start} ({type(price_start)}), End: {price_end} ({type(price_end)})")
                    
                    # Ensure price_start and price_end are correctly ordered
                    zone_high = max(price_start, price_end)
                    zone_low = min(price_start, price_end)
                    
                    # Enhanced POI status classification
                    if zone_type == 'supply':
                        if current_price > zone_high:
                            poi.status = 'broken'
                        elif current_price > zone_low and current_price < zone_high:
                            poi.status = 'tested'
                        else:
                            poi.status = 'active'
                    else:  # demand
                        if current_price < zone_low:
                            poi.status = 'broken'
                        elif current_price < zone_high and current_price > zone_low:
                            poi.status = 'tested'
                        else:
                            poi.status = 'active'
                except Exception as e:
                    logger.error(f"Error processing POI zone: {str(e)}")
    
    def _process_poi_results(self, poi_zones: Dict[str, List[POI]], current_price: float) -> Dict:
        """Process POI results and return the analysis."""
        try:
            # Get nearest POIs for immediate use and ensure numeric values
            try:
                # Safely filter supply zones above current price
                valid_supply_zones = []
                for p in poi_zones['supply']:
                    try:
                        p_start = float(p.price_start)
                        if p_start > current_price:
                            valid_supply_zones.append((p, p_start))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid price_start value in supply zone: {p.price_start}")
                
                # Find nearest supply zone above current price
                nearest_supply = None
                if valid_supply_zones:
                    nearest_supply = min(valid_supply_zones, key=lambda x: x[1])[0]
                
                # Safely filter demand zones below current price
                valid_demand_zones = []
                for p in poi_zones['demand']:
                    try:
                        p_end = float(p.price_end)
                        if p_end < current_price:
                            valid_demand_zones.append((p, p_end))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid price_end value in demand zone: {p.price_end}")
                
                # Find nearest demand zone below current price
                nearest_demand = None
                if valid_demand_zones:
                    nearest_demand = max(valid_demand_zones, key=lambda x: x[1])[0]
                
                # Calculate distance to nearest POIs
                distance_to_supply = float(nearest_supply.price_start) - current_price if nearest_supply else None
                distance_to_demand = current_price - float(nearest_demand.price_end) if nearest_demand else None
                
                # Get active zones (not broken) with safe type conversion
                active_zones = []
                for zone_type in ['supply', 'demand']:
                    for p in poi_zones[zone_type]:
                        if p.status != 'broken':
                            try:
                                # Use price_start for distance calculation, with fallback to price_end
                                p_val = float(p.price_start)
                                distance = abs(current_price - p_val)
                                active_zones.append({**vars(p), 'distance': distance})
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid price value in {zone_type} zone, skipping")
                
                # Sort active zones by distance to current price
                active_zones.sort(key=lambda z: z['distance'])
                
                # Get fresh and untested zones (higher quality)
                fresh_zones = [z for z in active_zones if z['status'] == 'active']
                
                # Get recently tested zones (potential reaction points)
                tested_zones = [z for z in active_zones if z['status'] == 'tested']
                
                # Safely extract numeric values
                resistance_level = None
                support_level = None
                
                if nearest_supply:
                    try:
                        resistance_level = float(nearest_supply.price_start)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid resistance level: {nearest_supply.price_start}")
                
                if nearest_demand:
                    try:
                        support_level = float(nearest_demand.price_end)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid support level: {nearest_demand.price_end}")
                
                logger.debug(f"POI Prices - Resistance: {resistance_level} ({type(resistance_level)}), "
                            f"Support: {support_level} ({type(support_level)})")
                
                result = {
                    'resistance': resistance_level,
                    'support': support_level,
                    'current_price': current_price,
                    'zones': {
                        'supply': [vars(p) for p in poi_zones['supply']],
                        'demand': [vars(p) for p in poi_zones['demand']]
                    },
                    'active_zones': active_zones,
                    'fresh_zones': fresh_zones,
                    'tested_zones': tested_zones,
                    'zone_counts': {
                        zone_type: len([
                            p for p in poi_zones[zone_type]
                            if p.status != 'broken'
                        ])
                        for zone_type in ['supply', 'demand']
                    },
                    'distance_to_supply': distance_to_supply,
                    'distance_to_demand': distance_to_demand
                }
                
                logger.debug(f"POI Result Structure: {list(result.keys())}")
                logger.debug(f"POI Result Types - current_price: {type(result['current_price'])}, "
                            f"resistance: {type(result['resistance']) if result['resistance'] else None}, "
                            f"support: {type(result['support']) if result['support'] else None}")
                
                return result
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Error processing nearest POIs: {str(e)}")
                logger.debug(f"Error trace: {error_trace}")
                
                # Return basic structure with empty values
                return {
                    'resistance': None,
                    'support': None,
                    'current_price': current_price,
                    'zones': {'supply': [], 'demand': []},
                    'active_zones': [],
                    'fresh_zones': [],
                    'tested_zones': [],
                    'zone_counts': {'supply': 0, 'demand': 0},
                    'distance_to_supply': None,
                    'distance_to_demand': None
                }
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error processing POI results: {str(e)}")
            logger.debug(f"Error trace: {error_trace}")
            
            # Return basic structure with empty values
            return {
                'resistance': None,
                'support': None,
                'current_price': current_price,
                'zones': {'supply': [], 'demand': []},
                'active_zones': [],
                'fresh_zones': [],
                'tested_zones': [],
                'zone_counts': {'supply': 0, 'demand': 0},
                'distance_to_supply': None,
                'distance_to_demand': None
            }
    
    def _get_empty_result(self, df: pd.DataFrame) -> Dict:
        """Return an empty result structure."""
        return {
            'resistance': None,
            'support': None,
            'current_price': float(df['close'].iloc[-1]) if not df.empty else 0,
            'zones': {'supply': [], 'demand': []},
            'active_zones': [],
            'fresh_zones': [],
            'tested_zones': [],
            'zone_counts': {'supply': 0, 'demand': 0},
            'distance_to_supply': None,
            'distance_to_demand': None
        }
        
    def check_poi_alignment(
        self,
        current_analysis: Dict,
        higher_tf_analysis: Dict,
        current_price: float,
        atr: Optional[float] = None
    ) -> bool:
        """Check if POIs are aligned across timeframes for stronger confirmation.
        
        Args:
            current_analysis: POI analysis from current timeframe
            higher_tf_analysis: POI analysis from higher timeframe
            current_price: Current market price
            atr: Average True Range for tolerance calculation
            
        Returns:
            bool: True if POIs are aligned or if alignment can't be determined
        """
        try:
            if current_price <= 0:
                logger.warning("Invalid current price in analysis")
                return True
                
            # Get trend direction
            trend = current_analysis.get('trend', 'neutral').lower()
            
            # Get POI levels
            current_supply = self._get_numeric_price(current_analysis.get('resistance'))
            current_demand = self._get_numeric_price(current_analysis.get('support'))
            higher_supply = self._get_numeric_price(higher_tf_analysis.get('resistance'))
            higher_demand = self._get_numeric_price(higher_tf_analysis.get('support'))
            
            # Calculate tolerance
            if not atr:
                atr = current_price * 0.003  # Default to 0.3% of price
            tolerance = atr * 2
            
            return self._check_alignment_by_trend(
                trend, current_price,
                current_supply, current_demand,
                higher_supply, higher_demand,
                tolerance, atr
            )
            
        except Exception as e:
            logger.error(f"Error checking POI alignment: {str(e)}")
            return True
            
    def _get_numeric_price(self, price_data: Any) -> Optional[float]:
        """Extract numeric price from various possible formats."""
        try:
            if isinstance(price_data, (int, float)):
                return float(price_data)
            elif isinstance(price_data, dict):
                if 'price_start' in price_data:
                    return float(price_data['price_start'])
                elif 'price_end' in price_data:
                    return float(price_data['price_end'])
            return None
        except (TypeError, ValueError):
            return None
            
    def _check_alignment_by_trend(
        self,
        trend: str,
        current_price: float,
        current_supply: Optional[float],
        current_demand: Optional[float],
        higher_supply: Optional[float],
        higher_demand: Optional[float],
        tolerance: float,
        atr: float
    ) -> bool:
        """Check POI alignment based on trend direction."""
        if trend == 'bullish':
            if current_demand is not None and higher_demand is not None:
                demand_aligned = abs(current_demand - higher_demand) <= tolerance
                near_demand = (current_price - current_demand) <= atr * 1.5
                logger.debug(f"Bullish POI Alignment: Demand zones aligned: {demand_aligned}, Near demand: {near_demand}")
                return demand_aligned or near_demand
            return current_demand is not None or higher_demand is not None
            
        elif trend == 'bearish':
            if current_supply is not None and higher_supply is not None:
                supply_aligned = abs(current_supply - higher_supply) <= tolerance
                near_supply = (current_supply - current_price) <= atr * 1.5
                logger.debug(f"Bearish POI Alignment: Supply zones aligned: {supply_aligned}, Near supply: {near_supply}")
                return supply_aligned or near_supply
            return current_supply is not None or higher_supply is not None
            
        return True  # Neutral trend or unknown