"""
Site Predictor - API Wrapper for RF Predictions
Uses existing rf_design_tool modules to predict site deployment impact
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import h3

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from rf_design_tool.modules.data_loader import DataLoader
from rf_design_tool.modules.prediction_engine import PredictionEngine


class SitePredictor:
    """
    Predicts coverage impact of deploying a new site
    Calculates bins with â‰¥3dB improvement vs current baseline
    """
    
    def __init__(self, baseline_file=None):
        """
        Initialize predictor with baseline network and ML model
        OPTIMIZED: Preloads models and geographic features for fast predictions
        
        Args:
            baseline_file: Path to baseline RSRP CSV (uses comprehensive_rsrp_all_46_sites.csv by default)
        """
        print("Initializing Site Predictor (OPTIMIZED MODE)...")
        
        # Set baseline file — prefer /tmp/ on Cloud Run (downloaded from GCS at startup)
        if baseline_file is None:
            tmp_baseline = Path('/tmp/comprehensive_rsrp_all_46_sites.csv')
            baseline_file = tmp_baseline if tmp_baseline.exists() else parent_dir / 'comprehensive_rsrp_all_46_sites.csv'
        
        # Load RF design tool modules
        print("  Loading models and data...")
        self.loader = DataLoader(str(parent_dir))

        # Load BOTH models so user can toggle without restart
        import joblib as _joblib, json as _json
        _base_model_path = parent_dir / 'Model' / 'lean_lgbm_53feat_model.joblib'
        _base_feat_path  = parent_dir / 'Model' / 'lean_lgbm_53feat_features.json'
        _min5_model_path = parent_dir / 'Model' / 'lean_lgbm_53feat_min5_model.joblib'
        _min5_feat_path  = parent_dir / 'Model' / 'lean_lgbm_53feat_min5_features.json'

        self._models = {}
        if _base_model_path.exists():
            self._models['standard'] = (_joblib.load(_base_model_path), _json.load(open(_base_feat_path)))
            print(f'  standard model loaded: {_base_model_path.name}')
        if _min5_model_path.exists():
            self._models['min5'] = (_joblib.load(_min5_model_path), _json.load(open(_min5_feat_path)))
            print(f'  min5    model loaded: {_min5_model_path.name}')

        # Default to standard model
        self.model, self.features = self._models.get('standard', next(iter(self._models.values())))
        
        # OPTIMIZATION: Preload environmental features into memory
        print("  Preloading environmental features for all bins...")
        self.env_features = self.loader.load_environmental_features()
        print(f"  âœ“ Preloaded features for {len(self.env_features):,} bins")

        # Preload DSM database for LoS ray-tracing at inference time
        print("  Preloading DSM database for LoS ray-tracing...")
        self.dsm_lookup = self.loader.load_dsm_database()
        if self.dsm_lookup:
            print(f"  âœ“ DSM database loaded: {len(self.dsm_lookup):,} H3 bins")
        else:
            print("  âš  DSM database not found â€” DSM LoS features will use precomputed values")
        
        # Load baseline network predictions
        print(f"  Loading baseline from: {baseline_file}")
        self.baseline = pd.read_csv(baseline_file)
        
        # Create fast lookup dictionary
        self.baseline_dict = dict(zip(
            self.baseline['h3_index'], 
            self.baseline['predicted_rsrp']
        ))
        
        print(f"  âœ“ Baseline loaded: {len(self.baseline_dict):,} bins")
        print(f"  âœ“ RSRP range: {self.baseline['predicted_rsrp'].min():.1f} to {self.baseline['predicted_rsrp'].max():.1f} dBm")
        print("Site Predictor ready! (All data preloaded for fast predictions)\n")
    
    def predict_site_deployment(self, site_lat, site_lon, site_height=None,
                                 radius_m=1000, h3_resolution=10,
                                 model_variant='standard', edt=6):
        """
        Predict what happens if we deploy a new site at this location

        Args:
            site_lat: Site latitude
            site_lon: Site longitude
            site_height: Antenna height in metres (None â†’ auto-detect from clutter)
            radius_m: Prediction radius in metres (default 1000m)
            h3_resolution: H3 resolution for prediction bins (9/10/11).
                           Chosen by the caller based on map zoom level.
                           Lower = fewer bins = faster. Default 10 (~1-2s).

        Returns:
            Dictionary with:
                - geojson: GeoJSON FeatureCollection of improved bins
                - stats: Impact statistics
        """
        # Switch model based on requested variant
        if model_variant in self._models:
            self.model, self.features = self._models[model_variant]
            print(f'  Using model: {model_variant}')
        else:
            print(f'  WARNING: model_variant={model_variant!r} not found, using standard')
            self.model, self.features = self._models.get('standard', next(iter(self._models.values())))

        # AUTO-DETECT HEIGHT: Use building height at clicked location
        if site_height is None:
            site_height = self._get_building_height(site_lat, site_lon)
            print(f"  Auto-detected building height: {site_height:.1f}m at location")
        # Initialize prediction engine if not already done
        if not hasattr(self, 'engine'):
            self.engine = PredictionEngine(self.model, self.features, self.loader)
        
        # Define standard 3-sector configuration
        # RS power = 43 dBm (hardcoded), MDT = 0 (hardcoded), EDT = user input
        RS_POWER = 43.0
        MDT      = 0
        print(f"  Site params: height={site_height:.1f}m, RS={RS_POWER}dBm, EDT={edt}°, MDT={MDT}°")
        sectors = [
            {
                'name': 'Sector 1',
                'azimuth': 0,
                'edt': edt,         # Electrical downtilt from user input
                'mdt': MDT,         # Mechanical downtilt (always 0)
                'rs_power': RS_POWER,  # RS power always 43 dBm
                'frequency': 2100,
                'bandwidth': 20
            },
            {
                'name': 'Sector 2',
                'azimuth': 120,
                'edt': edt,
                'mdt': MDT,
                'rs_power': RS_POWER,
                'frequency': 2100,
                'bandwidth': 20
            },
            {
                'name': 'Sector 3',
                'azimuth': 240,
                'edt': edt,
                'mdt': MDT,
                'rs_power': RS_POWER,
                'frequency': 2100,
                'bandwidth': 20
            }
        ]
        
        # Predict coverage for new site using existing prediction engine
        print(f"Running predictions for site at ({site_lat:.4f}, {site_lon:.4f}), H3 res {h3_resolution}...")
        predictions = self.engine.predict_site_coverage(
            site_lat=site_lat,
            site_lon=site_lon,
            site_height=site_height,
            sectors=sectors,
            radius_m=radius_m,
            measured_only=False,       # Predict on all bins
            h3_resolution=h3_resolution,
            dsm_lookup=self.dsm_lookup,  # Live DSM LoS ray-tracing
        )
        
        # Calculate improvements vs baseline
        improvements_df = self._calculate_improvements(predictions)
        
        # Convert to GeoJSON for map display (improved bins only, for improvement view)
        geojson = self._to_geojson(improvements_df)
        
        # Convert ALL footprint bins (new_site_rsrp > baseline) for full-prediction view
        footprint_geojson = self._footprint_to_geojson(improvements_df)
        
        # Calculate statistics
        improved_mask = improvements_df['improvement'] >= 3.0
        footprint_mask = improvements_df['new_site_rsrp'] > improvements_df['baseline_rsrp']
        stats = {
            'total_bins': len(improvements_df),
            'improved_bins': int(improved_mask.sum()),
            'mean_improvement': float(improvements_df['improvement'].mean()),
            'max_improvement': float(improvements_df['improvement'].max()),
            'site_footprint_bins': int(footprint_mask.sum())
        }
        
        print(f"  âœ“ Predictions complete:")
        print(f"    - Total bins predicted: {stats['total_bins']:,}")
        print(f"    - Improved (â‰¥3dB): {stats['improved_bins']:,}")
        print(f"    - Site footprint (best server): {stats['site_footprint_bins']:,} bins")
        print(f"    - Mean improvement: {stats['mean_improvement']:.1f} dB\n")
        
        return {
            'geojson': geojson,
            'footprint_geojson': footprint_geojson,
            'stats': stats,
            'site_height': site_height
        }
    
    def _get_building_height(self, lat, lon):
        """
        Get realistic antenna height from clutter data at clicked location.
        Uses intelligent algorithm to distinguish buildings from trees/poles:
        
        1. Building-dominant bin (Building >= 10 AND Building >= Tree):
           â†’ Use clutter_p95_height (filters out tree/pole spikes)
        2. Some buildings (Building > 0):
           â†’ Use clutter_mean_height (more conservative)
        3. No buildings:
           â†’ Fall back to clutter_max_height (might be pole/tree)
        
        Args:
            lat, lon: Site location
        
        Returns:
            Antenna height in meters
        """
        # Get H3 index at clicked location
        h3_idx = h3.latlng_to_cell(lat, lon, 12)
        
        if h3_idx not in self.env_features.index:
            print(f"  Warning: bin {h3_idx} at ({lat:.4f}, {lon:.4f}) not in env features â€” using default 20m")
            return 20.0
        
        bin_data = self.env_features.loc[h3_idx]
        
        # Get clutter metrics
        building_count = bin_data.get('Building_count', 0)
        tree_count = bin_data.get('Tree_count', 0)
        p95_height = bin_data.get('clutter_p95_height', None)
        mean_height = bin_data.get('clutter_mean_height', None)
        max_height = bin_data.get('clutter_max_height', None)
        
        # Determine best height estimate based on clutter composition
        if building_count >= 10 and building_count >= tree_count:
            # Building-dominant: p95 filters tree spikes, captures true roof height
            if p95_height is not None and not pd.isna(p95_height):
                antenna_height = float(p95_height)
                print(f"  Building-dominant bin: using p95_height = {antenna_height:.1f}m "
                      f"(Building={int(building_count)}, Tree={int(tree_count)})")
            else:
                antenna_height = float(mean_height) if mean_height is not None else float(max_height)
                print(f"  Building-dominant (p95 N/A): using fallback = {antenna_height:.1f}m")
        
        elif building_count > 0:
            # Some buildings: mean height is more conservative
            if mean_height is not None and not pd.isna(mean_height):
                antenna_height = float(mean_height)
                print(f"  Mixed bin: using mean_height = {antenna_height:.1f}m "
                      f"(Building={int(building_count)}, Tree={int(tree_count)})")
            else:
                antenna_height = float(max_height)
                print(f"  Mixed bin (mean N/A): using max_height = {antenna_height:.1f}m")
        
        else:
            # No buildings: might be vegetation/pole, use max
            if max_height is not None and not pd.isna(max_height):
                antenna_height = float(max_height)
                print(f"  No buildings: using max_height = {antenna_height:.1f}m (may be tree/pole)")
            else:
                raise ValueError(f"All clutter height metrics missing for bin {h3_idx}")
        
        return antenna_height
    
    def _get_baseline_rsrp(self, h3_idx: str) -> float:
        """
        Get baseline RSRP for an H3 bin at any resolution.
        - If the bin is res 12 (baseline native resolution), direct lookup.
        - If coarser (res 9-11), take the BEST (max) RSRP among all res-12
          children that exist in the baseline, representing the best server
          a UE in that area would see today.
        Returns -120 dBm if no baseline data found.
        """
        # Direct hit (works for res-12 and any res already in baseline)
        if h3_idx in self.baseline_dict:
            return self.baseline_dict[h3_idx]

        # Coarser resolution: find all res-12 children
        try:
            children = h3.cell_to_children(h3_idx, 12)
            rsrp_values = [self.baseline_dict[c] for c in children if c in self.baseline_dict]
            if rsrp_values:
                return max(rsrp_values)   # Best server a UE here would see
        except Exception:
            pass

        return -120.0   # Not covered

    def _calculate_improvements(self, predictions):
        """
        Calculate improvement for each bin vs baseline.
        Handles any H3 resolution via _get_baseline_rsrp().

        Args:
            predictions: DataFrame from prediction_engine with designed_rsrp

        Returns:
            DataFrame with improvement calculations
        """
        results = []

        for _, row in predictions.iterrows():
            h3_idx   = row['h3_index']
            new_rsrp = row['designed_rsrp']

            # Baseline RSRP â€” works for res 9/10/11/12
            baseline_rsrp = self._get_baseline_rsrp(h3_idx)

            final_rsrp  = max(baseline_rsrp, new_rsrp)
            improvement = final_rsrp - baseline_rsrp

            results.append({
                'h3_index':       h3_idx,
                'baseline_rsrp':  baseline_rsrp,
                'new_site_rsrp':  new_rsrp,
                'final_rsrp':     final_rsrp,
                'improvement':    improvement,
                'is_improved':    improvement >= 3.0,
                'serving_sector': row['serving_sector'],
                'sector_name':    row['sector_name']
            })

        return pd.DataFrame(results)
    
    def _calculate_site_footprint(self, predictions):
        """
        Calculate site footprint: bins where new site becomes best server
        
        This compares new site RSRP vs all existing sites to determine
        where the new site would actually serve customers
        """
        footprint = []
        
        for _, row in predictions.iterrows():
            h3_idx = row['h3_index']
            new_rsrp = row['designed_rsrp']
            baseline_rsrp = self.baseline_dict.get(h3_idx, -120)
            
            # New site wins if its RSRP is better than current best server
            if new_rsrp > baseline_rsrp:
                footprint.append(h3_idx)
        
        return footprint
    
    @staticmethod
    def _h3_to_polygon(h3_index):
        """Return GeoJSON Polygon geometry for an H3 cell (exterior ring, closed)."""
        boundary = h3.cell_to_boundary(h3_index)   # [(lat,lon), ...]
        # GeoJSON uses [lon, lat] order; close the ring by repeating the first point
        coords = [[round(lon, 6), round(lat, 6)] for lat, lon in boundary]
        coords.append(coords[0])
        return {'type': 'Polygon', 'coordinates': [coords]}

    def _footprint_to_geojson(self, improvements_df):
        """
        Convert ALL bins where new site is best server (new_site_rsrp > baseline)
        to GeoJSON Polygon (hex) features coloured by new_site_rsrp.
        Hexagons tile perfectly at any zoom level.
        """
        features = []
        footprint = improvements_df[improvements_df['new_site_rsrp'] > improvements_df['baseline_rsrp']]

        for _, row in footprint.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': self._h3_to_polygon(row['h3_index']),
                'properties': {
                    'new_site_rsrp': round(row['new_site_rsrp'], 1),
                    'baseline_rsrp': round(row['baseline_rsrp'], 1),
                    'improvement_db': round(row['improvement'], 1),
                    'sector_name': row['sector_name']
                }
            }
            features.append(feature)

        return {'type': 'FeatureCollection', 'features': features}

    def _to_geojson(self, improvements_df):
        """
        Convert ALL prediction bins to GeoJSON Polygon (hex) features.
        Improved bins (≥3dB) carry improvement_db > 0; others carry 0.
        Hexagons tile perfectly and are visible at all zoom levels.

        Args:
            improvements_df: DataFrame with improvement calculations

        Returns:
            GeoJSON FeatureCollection of Polygon features for ALL footprint bins
        """
        features = []

        for _, row in improvements_df.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': self._h3_to_polygon(row['h3_index']),
                'properties': {
                    'improvement_db':  round(row['improvement'], 1),
                    'final_rsrp':      round(row['final_rsrp'], 1),
                    'baseline_rsrp':   round(row['baseline_rsrp'], 1),
                    'new_site_rsrp':   round(row['new_site_rsrp'], 1),
                    'is_improved':     bool(row['is_improved']),
                    'serving_sector':  int(row['serving_sector']),
                    'sector_name':     row['sector_name']
                }
            }
            features.append(feature)

        return {'type': 'FeatureCollection', 'features': features}


# Test harness
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Site Predictor")
    print("="*60 + "\n")
    
    # Initialize
    predictor = SitePredictor()
    
    # Test prediction at a sample location in Windsor
    test_lat = 42.30
    test_lon = -83.05
    
    result = predictor.predict_site_deployment(
        site_lat=test_lat,
        site_lon=test_lon,
        site_height=30,
        radius_m=1000  # Small radius for testing
    )
    
    print("Test Results:")
    print(f"  GeoJSON features: {len(result['geojson']['features'])}")
    print(f"  Stats: {result['stats']}")
    print("\nâœ… Site Predictor test passed!")

