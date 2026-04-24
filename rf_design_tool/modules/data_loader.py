"""
Data Loader Module
Loads and prepares datasets for the RF Design Tool
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import h3

# ── Google Cloud Storage download ─────────────────────────────────────────────
# When REGION env var is set (per-city deployment), files are pre-downloaded at
# module import time — before gunicorn starts listening — so they're already in
# /tmp/ when the background init thread runs. This restores the fast startup
# behaviour from before multi-region support was added.
# When REGION is not set (local dev / multi-region mode), downloads are lazy.
# When GCS_BUCKET is not set (local dev), all downloads are skipped entirely.

_GCS_BUCKET = os.environ.get("GCS_BUCKET", "windsor-rf-ml-data")

_GCS_FILES_WINDSOR = {
    "h3_complete_features_windsor.csv":    "/tmp/h3_complete_features_windsor.csv",
    "h3_dsm_clutter_database.csv":         "/tmp/h3_dsm_clutter_database.csv",
    "h3_dem_database.csv":                 "/tmp/h3_dem_database.csv",
    "comprehensive_rsrp_all_46_sites.csv": "/tmp/comprehensive_rsrp_all_46_sites.csv",
    "dataset.csv":                         "/tmp/dataset.csv",
    "missing_sites_dataset.csv":           "/tmp/missing_sites_dataset.csv",
}

_GCS_FILES_MONTREAL = {
    "montreal/h3_complete_features_montreal.csv":          "/tmp/h3_complete_features_montreal.csv",
    "montreal/h3_dsm_database_montreal.csv":               "/tmp/h3_dsm_database_montreal.csv",
    "montreal/h3_dem_database_montreal.csv":               "/tmp/h3_dem_database_montreal.csv",
    "montreal/montreal_baseline_rsrp.csv":                 "/tmp/montreal_baseline_rsrp.csv",
    "montreal/lgbm_montreal_53feat_urban_model.joblib":    "/tmp/lgbm_montreal_53feat_urban_model.joblib",
    "montreal/lgbm_montreal_53feat_suburban_model.joblib": "/tmp/lgbm_montreal_53feat_suburban_model.joblib",
    "montreal/lgbm_montreal_53feat_urban_features.json":   "/tmp/lgbm_montreal_53feat_urban_features.json",
    "montreal/lgbm_montreal_53feat_suburban_features.json":"/tmp/lgbm_montreal_53feat_suburban_features.json",
    "montreal/urban_mtl.geojson":                          "/tmp/urban_mtl.geojson",
    "montreal/mtl_cells_1900_2100.csv":                    "/tmp/mtl_cells_1900_2100.csv",
}


def download_region_files(region_name: str):
    """Download data files for the selected region from GCS to /tmp/.
    No-op locally (GCS_BUCKET not set). Skips files already present."""
    if "GCS_BUCKET" not in os.environ:
        return
    files = _GCS_FILES_WINDSOR if region_name == 'windsor' else _GCS_FILES_MONTREAL
    label = region_name.capitalize()
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET)
        print(f"GCS: downloading {label} data files from gs://{_GCS_BUCKET}/")
        for gcs_name, local_path in files.items():
            if Path(local_path).exists():
                print(f"  ✓ Already present: {local_path}")
                continue
            blob = bucket.blob(gcs_name)
            print(f"  ↓ {gcs_name}")
            blob.download_to_filename(local_path)
            print(f"  ✓ Done: {local_path}")
        print(f"GCS: {label} files ready.\n")
    except Exception as e:
        print(f"GCS {label} download failed: {e}")
        raise


# Pre-download at import time for per-city deployments (REGION env var set)
_startup_region = os.environ.get('REGION', '').lower()
if _startup_region and os.environ.get('GCS_BUCKET'):
    download_region_files(_startup_region)


def _resolve_path(local_path: Path, tmp_name: str) -> Path:
    """
    Return /tmp/<tmp_name> if running on Cloud Run (file was downloaded from GCS),
    otherwise return the original local_path.
    """
    tmp_path = Path(f"/tmp/{tmp_name}")
    if tmp_path.exists():
        return tmp_path
    return local_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading all required datasets and models"""
    
    def __init__(self, base_path: str = None,
                 h3_features_path: str = None,
                 terrain_elevation_fallback: float = 183.0,
                 dsm_path: str = None,
                 dem_path: str = None):
        """
        Args:
            base_path: Base path to data directory. If None, uses parent directory.
            h3_features_path: Full path to H3 environmental features CSV.
            terrain_elevation_fallback: Default terrain elevation (m ASL).
            dsm_path: Full path to DSM CSV (h3_index, p95_height). Overrides default.
            dem_path: Full path to DEM CSV (h3_index, dem_mean). Overrides default.
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)
        self._h3_features_path_override = Path(h3_features_path) if h3_features_path else None
        self._dsm_path_override = Path(dsm_path) if dsm_path else None
        self._dem_path_override = Path(dem_path) if dem_path else None
        self.terrain_elevation_fallback = terrain_elevation_fallback
        
        logger.info(f"Data loader initialized with base path: {self.base_path}")
        
        # Cached data
        self._measured_rsrp = None
        self._environmental_features = None
        self._model = None
        self._antenna_patterns = None
        self._feature_names = None
        self._dsm_lookup = None   # h3_index -> mean_height (DSM, m ASL)
    
    def load_measured_rsrp(self, force_reload: bool = False) -> Dict[str, float]:
        """
        Load sparse measured RSRP data from dataset.csv
        
        Returns:
            Dictionary mapping h3_index -> rsrp_dbm for measured locations only
        """
        if self._measured_rsrp is not None and not force_reload:
            return self._measured_rsrp
        
        logger.info("Loading measured RSRP data...")
        
        # Load dataset (prefer /tmp/ on Cloud Run, fall back to local base_path)
        dataset_path = _resolve_path(self.base_path / "dataset.csv", "dataset.csv")

        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        
        # Extract measured RSRP by H3 index
        # Check which columns exist
        if 'h3_index' not in df.columns:
            logger.error("'h3_index' column not found in dataset")
            raise ValueError("Dataset must contain 'h3_index' column")
        
        if 'rsrp_dbm' not in df.columns:
            logger.error("'rsrp_dbm' column not found in dataset")
            raise ValueError("Dataset must contain 'rsrp_dbm' column")
        
        # Filter to valid H3 indices and RSRP values
        valid_data = df[['h3_index', 'rsrp_dbm']].dropna()
        
        # Create mapping (using mean if multiple measurements per bin)
        measured_rsrp = valid_data.groupby('h3_index')['rsrp_dbm'].mean().to_dict()
        
        self._measured_rsrp = measured_rsrp
        
        logger.info(f"Loaded {len(measured_rsrp):,} H3 bins with measured RSRP")
        logger.info(f"RSRP range: {min(measured_rsrp.values()):.1f} to {max(measured_rsrp.values()):.1f} dBm")
        
        return self._measured_rsrp
    
    def load_environmental_features(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load environmental features for ALL H3 bins in Windsor
        Uses complete augmented database with 1.7M bins
        
        Returns:
            DataFrame indexed by h3_index with all environmental features
        """
        if self._environmental_features is not None and not force_reload:
            return self._environmental_features
        
        logger.info("Loading environmental features from COMPLETE augmented database...")
        
        # Load H3 environmental features database.
        # If an override path was provided at construction, use it directly.
        # Otherwise prefer /tmp/ (GCS download on Cloud Run) then local Windsor path.
        if self._h3_features_path_override:
            complete_db_path = self._h3_features_path_override
        else:
            complete_db_path = _resolve_path(
                self.base_path / "h3_complete_features_windsor.csv",
                "h3_complete_features_windsor.csv"
            )

        if not complete_db_path.exists():
            logger.warning(f"Complete augmented database not found at {complete_db_path}")
            logger.warning("Falling back to dataset.csv (limited to measured bins)")
            dataset_path = _resolve_path(self.base_path / "dataset.csv", "dataset.csv")
            df = pd.read_csv(dataset_path)
        else:
            logger.info(f"Loading from: {complete_db_path}")
            df = pd.read_csv(complete_db_path)
        
        # Define environmental feature columns - INCLUDE ALL CLUTTER FEATURES
        env_feature_cols = [
            'clutter_min_height', 'clutter_max_height', 'clutter_mean_height',
            'clutter_median_height', 'clutter_p95_height',
            'clutter_height_range', 'clutter_std_height',  # CRITICAL: Don't skip these!
            'tree_count', 'max_tree_width_cm', 'mean_tree_width_cm',
            'sum_tree_diameters_cm', 'std_tree_width_cm', 'median_tree_width_cm',
            'tree_density_per_km2', 'biomass_index',
            'water_area_m2', 'water_coverage_pct', 'has_water',
            'indoor_proportion', 'outdoor_proportion',
            'fresnel_radius_m', 'fresnel_clearance_ratio', 'fresnel_obstruction',
            'fresnel_tree_obstruction'  # CRITICAL: Include this!
        ]
        
        # Add clutter class percentages if available
        clutter_class_cols = [col for col in df.columns if col.endswith('_pct') and 'class' not in col.lower()]
        
        # Check which columns actually exist
        available_cols = ['h3_index']
        
        # Add lat/lon columns (handle different naming conventions)
        if 'end_location_lat' in df.columns:
            available_cols.extend(['end_location_lat', 'end_location_lon'])
        elif 'lat' in df.columns:
            available_cols.extend(['lat', 'lon'])
        
        for col in env_feature_cols:
            if col in df.columns:
                available_cols.append(col)
        
        # Add clutter class columns
        for col in df.columns:
            if '_pct' in col and col not in available_cols:
                available_cols.append(col)
            if '_count' in col and 'point_count' not in col and col not in available_cols:
                available_cols.append(col)
        
        # Select unique H3 bins with their features
        env_features = df[available_cols].copy()
        
        # Group by H3 index and take mean (in case of duplicates)
        env_features = env_features.groupby('h3_index').mean().reset_index()
        
        # Fill NaN values with 0 for environmental features
        feature_cols = [col for col in env_features.columns if col not in ['h3_index', 'end_location_lat', 'end_location_lon']]
        env_features[feature_cols] = env_features[feature_cols].fillna(0)
        
        # Set index
        env_features = env_features.set_index('h3_index')
        
        self._environmental_features = env_features
        
        logger.info(f"Loaded environmental features for {len(env_features):,} H3 bins")
        logger.info(f"Feature columns: {len(env_features.columns)}")
        
        return self._environmental_features
    
    def load_model(self, force_reload: bool = False) -> Tuple[any, list]:
        """
        Load the trained LightGBM model and feature names
        
        Returns:
            Tuple of (model, feature_names)
        """
        if self._model is not None and not force_reload:
            return self._model, self._feature_names
        
        logger.info("Loading model...")
        
        # Prefer the lean 53-feature model; fall back to the original binned model
        lean_model_path   = self.base_path / "Model" / "lean_lgbm_53feat_model.joblib"
        lean_feature_path = self.base_path / "Model" / "lean_lgbm_53feat_features.json"
        legacy_model_path = self.base_path / "Model" / "lgbm_binned_model.joblib"

        if lean_model_path.exists():
            model_path = lean_model_path
            logger.info("Loading LEAN 53-feature model (lean_lgbm_53feat_model.joblib)")
        elif legacy_model_path.exists():
            model_path = legacy_model_path
            logger.warning("Lean model not found; falling back to lgbm_binned_model.joblib")
        else:
            raise FileNotFoundError(
                f"No model found. Expected:\n  {lean_model_path}\n  {legacy_model_path}"
            )

        self._model = joblib.load(model_path)

        # Feature names: prefer the explicit JSON list (lean model)
        if lean_feature_path.exists():
            import json
            with open(lean_feature_path) as f:
                self._feature_names = json.load(f)
            logger.info(f"Feature list loaded from JSON: {len(self._feature_names)} features")
        else:
            # Fall back to model introspection
            try:
                self._feature_names = self._model.feature_name_
            except AttributeError:
                try:
                    self._feature_names = self._model.booster_.feature_name()
                except Exception:
                    logger.warning("Could not extract feature names from model")
                    self._feature_names = None
        
        logger.info(f"Model loaded successfully")
        if self._feature_names:
            logger.info(f"Model expects {len(self._feature_names)} features")
        
        return self._model, self._feature_names
    
    def load_antenna_patterns(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load antenna pattern library
        
        Returns:
            DataFrame with antenna patterns
        """
        if self._antenna_patterns is not None and not force_reload:
            return self._antenna_patterns
        
        logger.info("Loading antenna patterns...")
        
        pattern_path = self.base_path / "antenna_patterns_merged.csv"
        
        if not pattern_path.exists():
            logger.error(f"Antenna patterns not found at {pattern_path}")
            raise FileNotFoundError(f"Antenna patterns not found: {pattern_path}")
        
        self._antenna_patterns = pd.read_csv(pattern_path)
        
        logger.info(f"Loaded {len(self._antenna_patterns)} antenna patterns")
        
        return self._antenna_patterns
    
    def get_h3_bins_in_radius(self, center_lat: float, center_lon: float,
                               radius_m: float = 2000, resolution: int = 12) -> list:
        """
        Get all H3 bins within a radius of a center point.

        Uses a polygon-based approach (h3.geo_to_cells) which is O(n_cells)
        instead of grid_disk which is O(k²) and extremely slow for large radii
        at fine resolutions.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_m: Radius in meters
            resolution: H3 resolution (default 12)

        Returns:
            List of H3 indices
        """
        import math

        # Convert radius to degrees (approximate, valid for small areas)
        # 1 degree lat ≈ 111,320m; longitude degrees vary with cos(lat)
        lat_deg = radius_m / 111320.0
        lon_deg = radius_m / (111320.0 * math.cos(math.radians(center_lat)))

        # Build a circle polygon with 32 vertices
        n_pts = 32
        coords = []
        for i in range(n_pts):
            angle = 2 * math.pi * i / n_pts
            lat = center_lat + lat_deg * math.sin(angle)
            lon = center_lon + lon_deg * math.cos(angle)
            coords.append([lat, lon])
        coords.append(coords[0])  # close the ring

        # h3.geo_to_cells expects GeoJSON-style polygon: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lat, lon in coords]]
        }

        cells = list(h3.geo_to_cells(geojson_polygon, resolution))

        # ── Filter out water bins (river / lake) using environmental features ──
        # Bins with water_coverage_pct > 90 are predominantly open water
        # (Detroit River, Lake St. Clair, Lake Erie) — skip them for prediction.
        env = self.load_environmental_features()
        if 'water_coverage_pct' in env.columns:
            water_set = set(env.index[env['water_coverage_pct'] > 90])
            before = len(cells)
            cells = [c for c in cells if c not in water_set]
            removed = before - len(cells)
            if removed > 0:
                logger.info(f"  Removed {removed} water bins (water_coverage_pct > 90)")
        elif 'has_water' in env.columns:
            # Fallback: use has_water flag
            water_set = set(env.index[env['has_water'] == 1])
            before = len(cells)
            cells = [c for c in cells if c not in water_set]
            removed = before - len(cells)
            if removed > 0:
                logger.info(f"  Removed {removed} water bins (has_water == 1)")

        logger.info(f"Found {len(cells):,} H3 res-{resolution} bins within {radius_m}m of ({center_lat:.4f}, {center_lon:.4f})")
        return cells
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def load_dsm_database(self, force_reload: bool = False) -> Dict[str, dict]:
        """
        Load H3-indexed DSM + DEM databases for LoS ray-tracing.

        Combines two sources (all heights in metres ASL):
          h3_dem_database.csv      — bare-ground terrain (HRDEM-Terrain.tif)
                                     dem_mean used for antenna ground elevation:
                                     ant_asl = dem_mean(ant_bin) + ant_height_agl
          h3_dsm_clutter_database.csv — DSM surface including buildings/trees
                                     p95_height used for ray obstruction checking

        Returns:
            Dict: h3_index -> {'dem': float (terrain ASL), 'p95': float (surface ASL)}
        """
        if self._dsm_lookup is not None and not force_reload:
            return self._dsm_lookup

        dem_path = self._dem_path_override or _resolve_path(self.base_path / "h3_dem_database.csv", "h3_dem_database.csv")
        dsm_path = self._dsm_path_override or _resolve_path(self.base_path / "h3_dsm_clutter_database.csv", "h3_dsm_clutter_database.csv")

        if not dsm_path.exists():
            logger.warning(f"DSM database not found at {dsm_path}. DSM LoS features will default to 0.")
            self._dsm_lookup = {}
            return self._dsm_lookup

        # ── Load DSM p95 surface heights ──────────────────────────────────────
        logger.info(f"Loading DSM p95 surface heights from: {dsm_path}")
        dsm_df = pd.read_csv(dsm_path, usecols=["h3_index", "p95_height"])
        dsm_df = dsm_df.set_index("h3_index")
        p95_dict = dsm_df["p95_height"].astype(float).to_dict()

        # ── Load DEM bare-ground terrain elevations ───────────────────────────
        dem_dict = {}
        if dem_path.exists():
            logger.info(f"Loading DEM bare-ground elevations from: {dem_path}")
            dem_df = pd.read_csv(dem_path, usecols=["h3_index", "dem_mean"])
            dem_df = dem_df.set_index("h3_index")
            dem_dict = dem_df["dem_mean"].astype(float).to_dict()
            logger.info(f"DEM loaded: {len(dem_dict):,} H3 bins")
        else:
            logger.warning(f"DEM database not found at {dem_path}. "
                           f"Will fall back to Windsor default terrain (183m ASL).")

        # ── Merge: all DSM bins, add DEM where available ──────────────────────
        WINDSOR_TERRAIN_ASL = self.terrain_elevation_fallback
        self._dsm_lookup = {
            idx: {
                "dem": dem_dict.get(idx, WINDSOR_TERRAIN_ASL),
                "p95": p95,
            }
            for idx, p95 in p95_dict.items()
        }

        n_dem = sum(1 for idx in p95_dict if idx in dem_dict)
        logger.info(f"DSM+DEM database ready: {len(self._dsm_lookup):,} H3 bins "
                    f"({n_dem:,} with real DEM, rest use {WINDSOR_TERRAIN_ASL}m fallback)")
        return self._dsm_lookup

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of loaded data"""
        stats = {}
        
        if self._measured_rsrp:
            stats['measured_bins'] = len(self._measured_rsrp)
            rsrp_values = list(self._measured_rsrp.values())
            stats['rsrp_min'] = min(rsrp_values)
            stats['rsrp_max'] = max(rsrp_values)
            stats['rsrp_mean'] = np.mean(rsrp_values)
        
        if self._environmental_features is not None:
            stats['env_feature_bins'] = len(self._environmental_features)
            stats['env_feature_count'] = len(self._environmental_features.columns)
        
        if self._model is not None:
            stats['model_loaded'] = True
            if self._feature_names:
                stats['model_features'] = len(self._feature_names)
        
        if self._antenna_patterns is not None:
            stats['antenna_patterns_count'] = len(self._antenna_patterns)
        
        return stats


# Convenience function
def create_data_loader(base_path: str = None) -> DataLoader:
    """Create and return a DataLoader instance"""
    return DataLoader(base_path)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    print("\n" + "="*60)
    print("Testing Data Loader")
    print("="*60)
    
    # Load measured RSRP
    measured = loader.load_measured_rsrp()
    print(f"\n✓ Measured RSRP: {len(measured):,} bins")
    
    # Load environmental features
    env = loader.load_environmental_features()
    print(f"✓ Environmental features: {len(env):,} bins, {len(env.columns)} features")
    
    # Load model
    model, features = loader.load_model()
    print(f"✓ Model loaded with {len(features) if features else '?'} features")
    
    # Test H3 radius search
    test_lat, test_lon = 42.3, -83.0
    bins = loader.get_h3_bins_in_radius(test_lat, test_lon, 2000)
    print(f"✓ H3 bins in 2km radius: {len(bins):,}")
    
    # Summary
    stats = loader.get_summary_stats()
    print("\nSummary Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ All tests passed!")
